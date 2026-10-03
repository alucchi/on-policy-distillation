"""Generate answers on AIME24 and the released LiveCodeBench v5 subset."""

import argparse
import json
import os
import random
from pathlib import Path

from common import MODELS, load_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODELS["student"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0, help="Problems per benchmark; 0 means all")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", nargs="+", default=["aime24", "livecodebench_v5"],
                        choices=["aime24", "livecodebench_v5"])
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    # These BF16 models do not need optional DeepGEMM FP8 kernels.
    os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")
    from vllm import LLM, SamplingParams

    engine = LLM(
        model=args.model, dtype=args.dtype, seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len, generation_config="vllm",
    )
    tokenizer = engine.get_tokenizer()
    context_limit = engine.llm_engine.model_config.max_model_len
    output.mkdir(parents=True, exist_ok=True)
    (output / "eval_config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    temperatures = {"aime24": 0.6, "livecodebench_v5": 1.0}
    for benchmark in args.benchmarks:
        temperature = temperatures[benchmark]
        dataset = load_prompts("eval_" + benchmark)
        # Deduplicate legacy repeat rows, then use a fixed seeded problem subset.
        rows = list({str(row["sample_id"]): row for row in dataset}.values())
        random.Random(args.seed).shuffle(rows)
        rows = rows[:args.limit] if args.limit else rows
        generation = SamplingParams(
            n=args.samples, max_tokens=args.max_new_tokens,
            temperature=temperature, top_p=0.95, top_k=-1, seed=args.seed,
        )
        handle = (output / f"{benchmark}.jsonl").open("w")
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = []
            for row in batch:
                token_ids = tokenizer.apply_chat_template(
                    row["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False,
                )
                if len(token_ids) + args.max_new_tokens > context_limit:
                    raise ValueError(f"Prompt {row['sample_id']} plus generation exceeds context window")
                prompts.append({"prompt_token_ids": token_ids})
            # vLLM returns requests in input order, even when they finish out of order.
            results = engine.generate(prompts, generation, use_tqdm=True)
            if len(results) != len(batch):
                raise RuntimeError("vLLM returned an incomplete batch")
            for index, (row, result) in enumerate(zip(batch, results), start + 1):
                if len(result.outputs) != args.samples:
                    raise RuntimeError(f"Incomplete samples for {row['sample_id']}")
                for completion in result.outputs:
                    record = {
                        "sample_id": str(row["sample_id"]), "sample": completion.index,
                        "completion": completion.text,
                        "answer": row.get("answer"), "metadata": row["metadata"],
                        "tokens": len(completion.token_ids),
                        "hit_token_limit": completion.finish_reason == "length",
                    }
                    handle.write(json.dumps(record) + "\n")
                    handle.flush()
                print(f"{benchmark}: {index}/{len(rows)}", flush=True)
        handle.close()


if __name__ == "__main__":
    main()
