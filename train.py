"""Basic domain-routed multi-teacher on-policy distillation."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from common import MODELS, load_prompts


class VLLMRollouts:
    """Student rollouts from a vLLM engine process kept in sync with the trained weights."""

    def __init__(self, model, seed, max_model_len, gpu_memory_utilization, sync_dir):
        # vLLM serves SmolLM3 through its Transformers backend, which patches the HF model
        # classes it instantiates. Keep the engine in its own process (vLLM's default) so the
        # HF student and teachers in this process are untouched. It uses the first visible GPU.
        os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
        # Lets collective_rpc ship _load_weights to our own local engine process.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
        from vllm import LLM

        self.engine = LLM(
            model=model, dtype="bfloat16", seed=seed, max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization, generation_config="vllm",
            enable_prefix_caching=False,
        )
        self.sync_path = Path(sync_dir) / f"student-{os.getpid()}.safetensors"

    def generate(self, prompt_ids, max_new_tokens, seed):
        from vllm import SamplingParams

        # Plain temperature-1 sampling: the rollouts must come from the student distribution.
        params = SamplingParams(
            max_tokens=max_new_tokens, temperature=1.0, top_p=1.0, top_k=-1, seed=seed,
        )
        results = self.engine.generate(
            [{"prompt_token_ids": ids} for ids in prompt_ids], params, use_tqdm=False,
        )
        return [list(result.outputs[0].token_ids) for result in results]

    @torch.no_grad()
    def load_weights(self, model):
        from safetensors.torch import save_file

        # Hand BF16 weights to the engine process through a (RAM-backed) safetensors file.
        weights = {name: tensor.detach().to(torch.bfloat16).contiguous()
                   for name, tensor in model.state_dict().items()
                   if name != "lm_head.weight" or not model.config.tie_word_embeddings}
        save_file(weights, self.sync_path)
        self.engine.collective_rpc(_load_weights, args=(str(self.sync_path),))

    def close(self):
        self.sync_path.unlink(missing_ok=True)


def _load_weights(worker, path):
    from safetensors.torch import load_file

    worker.model_runner.get_model().load_weights(load_file(path).items())


def response_log_probs(model, tokens, prompt_length):
    # Full-vocabulary log-probs at each response position, shape [response_len, vocab].
    # Position prompt_length - 1 predicts the first response token; include EOS.
    logits = model(
        input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False,
        logits_to_keep=tokens.shape[1] - prompt_length + 1,
    ).logits[0, :-1].float()
    return logits.log_softmax(-1)


def sampled_log_probs(log_probs, tokens, prompt_length):
    return log_probs.gather(1, tokens[0, prompt_length:, None]).squeeze(1)


def opd_loss(student_logp, teacher_logp, reduction="mean"):
    # Exact per-token reverse KL(student || teacher) over the vocabulary, on the positions
    # of an on-policy student rollout. Unlike the single-sampled-token estimator, this
    # raises teacher-preferred tokens the student rarely samples (e.g. the opening
    # <think>) and has no high-variance single-token spikes.
    kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(-1)
    return kl.sum() if reduction == "sum" else kl.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", default=MODELS["student"])
    parser.add_argument("--math-teacher", default=MODELS["math"])
    parser.add_argument("--code-teacher", default=MODELS["code"])
    parser.add_argument("--output", default="outputs/student")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rollouts", type=int, default=8, help="Rollouts per domain per step")
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--student-device", default="cuda:1",
                        help="The vLLM rollout engine uses the first visible GPU (cuda:0)")
    parser.add_argument("--teacher-device", default="cuda:1")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--sync-dir", default="/dev/shm",
                        help="Directory for the per-step weight handoff file to vLLM")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.steps, args.rollouts, args.max_new_tokens, args.max_prompt_tokens) < 1 or args.lr <= 0:
        parser.error("steps, rollouts, token limits, and lr must be positive")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "mopd-distillation"),
        name=f"{output.name}-train",
        job_type="train",
        config=vars(args),
    )
    # cuDNN SDPA cannot build plans for some SmolLM3 training shapes.
    # Keep PyTorch flash/memory-efficient/math SDPA backends available.
    torch.backends.cuda.enable_cudnn_sdp(False)
    set_seed(args.seed)
    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    dataset = load_prompts("rl_prompt_mix")
    prompts = {"math": [], "code": []}
    for row in dataset:
        if row["domain"] in prompts:
            ids = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True, return_dict=False)
            if len(ids) <= args.max_prompt_tokens:
                prompts[row["domain"]].append(list(ids))
    if not all(prompts.values()):
        raise ValueError("No usable math/code prompts; increase --max-prompt-tokens")
    print({domain: len(rows) for domain, rows in prompts.items()}, flush=True)
    run.summary.update({domain + "_training_prompts": len(rows) for domain, rows in prompts.items()})

    rollouts = VLLMRollouts(args.student, args.seed, args.max_prompt_tokens + args.max_new_tokens,
                            args.vllm_gpu_memory_utilization, args.sync_dir)
    # FP32 parameters/Adam states preserve small updates; compute uses BF16 on CUDA.
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32)
    student.to(args.student_device)
    student.gradient_checkpointing_enable()
    for module in student.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    teachers = {}
    for domain, name in [("math", args.math_teacher), ("code", args.code_teacher)]:
        teacher_tokenizer = AutoTokenizer.from_pretrained(name)
        if (teacher_tokenizer.get_vocab() != tokenizer.get_vocab()
                or teacher_tokenizer.all_special_ids != tokenizer.all_special_ids):
            raise ValueError(f"{domain} teacher and student must share a tokenizer")
        teachers[domain] = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
        teachers[domain].to(args.teacher_device).eval().requires_grad_(False)
    context_limit = min(m.config.max_position_embeddings for m in [student, *teachers.values()])
    if args.max_prompt_tokens + args.max_new_tokens > context_limit:
        raise ValueError(f"Prompt + response limits exceed context length {context_limit}")
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    autocast = dict(device_type=student.device.type, dtype=torch.bfloat16,
                    enabled=student.device.type == "cuda")

    log = (output / "train.jsonl").open("w")
    for step in range(1, args.steps + 1):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = {"step": step}
        # One batched vLLM call samples every domain's rollouts from the current student.
        batch = [(domain, rng.choice(prompts[domain]))
                 for domain in teachers for _ in range(args.rollouts)]
        responses = rollouts.generate([ids for _, ids in batch], args.max_new_tokens,
                                      args.seed + step)
        generated = time.perf_counter()
        student.train()
        for domain, teacher in teachers.items():
            items = [(ids, response) for (d, ids), response in zip(batch, responses)
                     if d == domain and response]
            # Token mean within each domain; each domain gets equal update weight.
            total_tokens = sum(len(response) for _, response in items)
            kl_sum = sampled_kl_sum = 0.0
            for ids, response in items:
                tokens = torch.tensor([ids + response], device=args.student_device)
                with torch.no_grad():
                    teacher_logp = response_log_probs(
                        teacher, tokens.to(args.teacher_device), len(ids),
                    ).to(args.student_device)
                with torch.autocast(**autocast):
                    student_logp = response_log_probs(student, tokens, len(ids))
                    # Reverse KL D_KL(student || teacher), summed over response tokens.
                    # This is exact over the full vocabulary at each generated position.
                    loss = opd_loss(student_logp, teacher_logp, "sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at step {step}: {domain}")
                # Backpropagate the normalized reverse-KL objective; gradients accumulate
                # across rollouts before the optimizer step below.
                (loss / total_tokens / len(teachers)).backward()
                kl_sum += loss.item()
                sampled_kl_sum += (
                    sampled_log_probs(student_logp.detach(), tokens, len(ids))
                    - sampled_log_probs(teacher_logp, tokens, len(ids))
                ).sum().item()
                del student_logp, teacher_logp, loss, tokens
            metrics[domain + "_kl"] = kl_sum / max(total_tokens, 1)
            metrics[domain + "_sampled_kl"] = sampled_kl_sum / max(total_tokens, 1)
            metrics[domain + "_tokens"] = total_tokens / max(len(items), 1)
            metrics[domain + "_hit_token_limit"] = sum(
                response[-1] != tokenizer.eos_token_id for _, response in items
            ) / max(len(items), 1)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        rollouts.load_weights(student)
        metrics["grad_norm"] = grad_norm.item()
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        metrics["generation_seconds"] = generated - started
        metrics["step_seconds"] = time.perf_counter() - started
        metrics["response_tokens_per_second"] = sum(map(len, responses)) / metrics["step_seconds"]
        run.log(metrics, step=step)
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(json.dumps(metrics), flush=True)

    log.close()
    rollouts.close()
    student.save_pretrained(output)
    tokenizer.save_pretrained(output)
    run.summary["checkpoint"] = str(output.resolve())
    run.finish()


if __name__ == "__main__":
    main()
