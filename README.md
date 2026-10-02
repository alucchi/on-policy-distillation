# Math + code on-policy distillation

A minimal full-parameter PyTorch/Transformers implementation using:

- Student initialization: `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT`
- Math teacher: `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Math`
- Code teacher: `BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-RL-Code`

`run.sh` evaluates all three models on both benchmarks, trains one student, then
evaluates that student and prints a comparison table. Checkpoints, generated
answers, per-problem results, and scores are saved under `outputs/`.

## Setup

Use Python 3.10 or newer and a CUDA-enabled PyTorch installation. All generation uses vLLM. Training needs two GPUs by default: a vLLM engine on
`cuda:0` samples the student rollouts, and the student and both teachers train/score on
`cuda:1` (about 80 GB at 4,096-token rollouts). The student uses FP32 parameters and
optimizer states with BF16 computation and gradient checkpointing; the teachers use
BF16. After every optimizer step the BF16 student weights are handed to the vLLM
engine through a safetensors file in `/dev/shm` (`--sync-dir`). Evaluation uses vLLM
to load one BF16 model at a time and batch answer generation.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p third_party
git clone https://github.com/LiveCodeBench/LiveCodeBench third_party/LiveCodeBench
git -C third_party/LiveCodeBench checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
export PYTHONPATH="$PWD/third_party/LiveCodeBench${PYTHONPATH:+:$PYTHONPATH}"
```

If this machine reports that `ensurepip` is unavailable, create the environment
with `python3 -m venv --without-pip .venv` and install dependencies with
`python3 -m pip --python .venv/bin/python install -r requirements.txt`.

`score.py` automatically finds the checkout in `third_party/LiveCodeBench`,
including when invoked directly. `run.sh` checks scoring dependencies before
starting generation. You can check them manually with:

```bash
python score.py --check-dependencies
```

If generation completed before a scoring error, keep the saved answers and rerun
only scoring with the training/scoring interpreter (adjust the output path):

```bash
python score.py --input outputs/MixSFT
```

Only the official LiveCodeBench verifier is imported; its API-client and vLLM
packages are unnecessary for scoring. **The verifier executes generated Python. Run scoring
(and therefore `run.sh`) in a disposable, isolated environment without credentials
or sensitive files. Its subprocess timeouts are not a security sandbox.**

## Run

Training and evaluation share one environment with pinned PyTorch 2.11.0,
Transformers 5.5.4, and vLLM 0.20.0. With that environment activated:

```bash
bash run.sh
```

`PYTHON` selects the interpreter for all stages. `EVAL_PYTHON` optionally overrides
it for evaluation; it defaults to `PYTHON`. Older vLLM versions can fail with
`TokenizersBackend has no attribute all_special_tokens_extended` when mixed with
Transformers 5.

`run.sh` generates the three baselines in parallel, one GPU each, on the GPUs listed in
`EVAL_GPUS` (default `0,1,2`; physical GPU ids passed as `CUDA_VISIBLE_DEVICES`). With
fewer GPUs than models, models are assigned round-robin and run one after another on
their GPU, e.g. `EVAL_GPUS=0 bash run.sh` for a single GPU. Each model's generation
output goes to `$OUTPUT/logs/eval_<model>.log`; scoring and training start only after
all three succeed. The distilled student is evaluated on GPU 0 as before.

Evaluation submits up to 256 problems per batch. Saved answers and the
`aime24: N/30` counter update after each batch returns; vLLM displays generation
progress while the batch runs. Use `--batch-size 1` for per-problem updates.
`--tensor-parallel-size`, `--gpu-memory-utilization`, and `--max-model-len` control
engine resources. Select evaluation GPUs with `CUDA_VISIBLE_DEVICES` instead of
`--device`. Output JSONL and scoring remain compatible with previous runs.


Defaults: 100 optimizer steps, eight math and eight code rollouts per step, learning
rate `2e-6`, up to 4,096 generated training tokens, and 16 evaluation answers per
problem (`SAMPLES`) with up to 32,768 generated tokens. With 4,096 tokens nearly every AIME
answer is truncated before `</think>`, so even the teachers score ~0%. Training skips prompts longer than
2,048 tokens rather than truncating the problem. These are pilot settings, not a
claim of convergence or improved benchmark performance.

For a smaller end-to-end smoke run:

```bash
OUTPUT=outputs/smoke LIMIT=2 STEPS=2 TRAIN_TOKENS=64 EVAL_TOKENS=128 bash run.sh
```

The vLLM rollout engine always uses the first visible GPU. To colocate everything on
one GPU, lower the engine's share of memory:

```bash
python train.py --student-device cuda:0 --teacher-device cuda:0 --vllm-gpu-memory-utilization 0.3
```

You can also run each stage separately; `--help` lists the small set of options:

```bash
python generate.py --model BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-MixSFT --output outputs/baseline
python score.py --input outputs/baseline
python train.py --output outputs/student --steps 100 --lr 2e-6
python generate.py --model outputs/student --output outputs/final
python score.py --input outputs/final
```

`run.sh` reuses generation when `aime24.jsonl`, `livecodebench_v5.jsonl`, and
`eval_config.json` already exist in a model's output directory, then reruns scoring.
Partial file sets stop the run without overwriting answers; the scorer rejects any
run with incomplete samples. Cached generation retains its original settings;
choose a new `OUTPUT` directory to generate with different settings. This does not
resume training.

Use identical evaluation arguments before and after training. Output directories
must be empty for generation/training to prevent accidental overwrites. Training
saves the final model; this minimal implementation does not resume optimizer state.

## Weights & Biases

W&B logging is enabled by default for training only. Authenticate once after installing dependencies:

```bash
wandb login
WANDB_PROJECT=mopd-distillation WANDB_RUN_GROUP=math-code-01 bash run.sh
```

`WANDB_ENTITY` optionally selects your account/team. `run.sh` generates a unique
experiment group if `WANDB_RUN_GROUP` is unset. Training creates a `student-train`
run within that group. For separate training invocations, export
`WANDB_RUN_GROUP` yourself to associate them.

Training logs math/code exact and sampled KL, response lengths, token-limit
hits, gradient norm before clipping, learning rate, step time, and response-token
throughput. W&B also records training configuration and system metrics.
Generation and scoring do not initialize or log to W&B. Generation progress is
printed to the terminal; benchmark results are printed and saved in `scores.json`
and per-problem detail files. Checkpoints and generated text stay in local files.
KL to the teacher is not an accuracy metric; use
benchmark scores to assess capability improvements.

For [offline logging](https://docs.wandb.ai/support/models/articles/how-do-i-deal-with-network-issues)
or to disable W&B:

```bash
WANDB_MODE=offline bash run.sh
WANDB_MODE=disabled bash run.sh
```

Offline runs are saved under `wandb/` (or `WANDB_DIR`) and can later be uploaded
with `wandb sync wandb/offline-run-<id>`. Normal local JSON logs are still written.

## Training and evaluation details

Training uses only the math/code rows of the released `rl_prompt_mix` dataset.
Each step, vLLM samples fresh responses from the current student at temperature 1
without top-k or top-p filtering. The matching specialist scores every position of that response.
The loss is the exact per-token reverse KL, `mean_t sum_v p_student(v) *
(log p_student(v) - log p_teacher(v))`, over the full vocabulary at each response
position. Prompt tokens are excluded and EOS is included. Each domain's token mean
gets half the update weight, and both rollouts precede the optimizer step. Teachers
are frozen. `*_kl` is this exact KL; `*_sampled_kl` is the noisy single-token
estimate on the sampled tokens (it can be negative).

An earlier version optimized only the sampled-token surrogate
`stop_gradient(logp_student - logp_teacher) * logp_student` at lr `1e-5`. That
estimator only pushes down tokens the student samples, never directly raises
teacher-preferred tokens, and gives single tokens advantages of 15–25 nats. In
practice the student lost its opening `<think>` token (P went from 0.9999 to
~0) within 100 steps and evaluation collapsed.

This is a basic domain-routed M-OPD baseline. It omits Open-MOPD's adaptive domain
budgets, multi-update PPO machinery, and instruction-following teacher.

Evaluation uses the released 30 AIME24 problems and 167-problem LiveCodeBench v5
subset from [Open-MOPD Data](https://huggingface.co/datasets/BytedTsinghua-SIA/Open-MOPD-Data).
The training mixture is separate; the data card documents exclusion of
LiveCodeBench from code training. Original chat prompts are preserved.
AIME requires a final boxed integer (or an answer consisting solely of an integer).
LiveCodeBench uses its official code extraction and public/private test execution.
This AIME parser is stricter than the upstream fallback that searches arbitrary
numbers in the answer.

Scores are fractions in `scores.json`: AIME answer accuracy and LiveCodeBench
pass@1, averaged over samples, not “any sample passed.” `--limit` selects a seeded
subset and `--samples` controls repetitions. `run.sh` uses 16 samples (`SAMPLES`);
`generate.py` on its own defaults to one. Short outputs, single samples and small subsets are useful for smoke checks but produce noisy scores.
`hit_token_limit_fraction` reports how often generation stopped without EOS.

Math uses temperature 0.6, code 1.0, both with top-p 0.95. The published
[Open-MOPD protocol](https://github.com/BytedTsinghua-SIA/Open-MOPD) uses much longer
contexts and 64 math / 10 code samples; these pilot results are not directly
comparable. Increase token limits and sample counts for more reliable evaluation.

## Results

The student was trained with `bash run.sh` defaults (100 steps, 8 math + 8 code
rollouts per step, lr `2e-6`, 4,096 training tokens). All four models were then
evaluated with 16 samples per problem, up to 32,768 generated tokens and seed 42. The
AIME24 and LiveCodeBench columns can be reproduced with the code in this repository:

```bash
python generate.py --model <model> --output outputs_samples16/<name> --samples 16
python score.py --input outputs_samples16/<name>
```

The AIME25 column came from a modified `generate.py` / `score.py` that also loads the
`eval_aime25` prompts from `BytedTsinghua-SIA/Open-MOPD-Data` and scores them like
AIME24. That code is not part of this repository.

Scores are the mean correctness over all 16 samples (avg@16). Brackets give 95%
bootstrap intervals over problems; with only 30 problems per AIME set, the AIME
intervals are wide.

| Model | AIME24 | AIME25 | AIME24+25 (60) | LCB v5 |
|---|---:|---:|---:|---:|
| MixSFT (student init) | 16.0 | 20.2 | 18.1 [11.4, 25.6] | 15.7 [11.5, 20.0] |
| RL-Math (math teacher) | 23.5 | 26.7 | **25.1** [16.6, 34.2] | 18.0 [13.5, 22.7] |
| RL-Code (code teacher) | 20.6 | 21.9 | 21.2 [13.9, 29.4] | **23.5** [18.2, 28.9] |
| **Distilled student** | 24.6 | 25.6 | **25.1** [16.6, 34.2] | 22.5 [17.3, 27.8] |

Paired differences in points, comparing models on the same problems (bold: the 95%
interval excludes zero):

| Comparison | AIME24+25 | LCB v5 |
|---|---:|---:|
| RL-Math − RL-Code | **+3.9** [+0.8, +7.2] | **−5.5** [−8.5, −2.7] |
| Distilled − MixSFT | **+7.0** [+3.9, +10.6] | **+6.8** [+3.6, +10.3] |
| Distilled − RL-Math | +0.0 [−2.1, +2.1] | **+4.5** [+1.5, +7.6] |
| Distilled − RL-Code | **+3.9** [+0.9, +7.1] | −1.0 [−2.6, +0.4] |

- **Each teacher wins its own domain.** RL-Math leads on AIME and RL-Code on
  LiveCodeBench. An earlier one-sample run had RL-Code ahead on AIME24 (7/30 vs 6/30);
  that was noise.
- **The distilled student matches the better teacher in each domain:** tied with
  RL-Math on AIME and within one point of RL-Code on LiveCodeBench. It improves on
  MixSFT by about 7 points on both.
- **Code RL also helps math.** RL-Code is +3.1 points [+0.2, +6.2] over MixSFT on
  AIME. About 2 points of that comes from fewer truncated answers: MixSFT hits the
  32,768-token limit on 15–20% of AIME answers, RL-Code on 6–9%.
- **These numbers match the published model cards:** MixSFT 15.63 / 20.26 (AIME24 /
  AIME25) and 15.99 (LCB v5); RL-Math 23.65 / 24.84; RL-Code 22.16 on LCB v5.

During training the exact reverse KL to the teachers dropped from 0.014 (math) / 0.051
(code), averaged over the first 10 steps, to 0.003 / 0.004 over the last 10. Most
training rollouts still hit the 4,096-token limit at the end of training.

Scoring checks:
- AIME: no unparsed answer contained the correct number.
- LiveCodeBench: re-run with a 24 s limit instead of 6 s, only 13 of the 340
  time-limit failures in RL-Code and RL-Math pass. The distilled student writes
  unfenced code in 87 of its 2,672 answers; scoring those as code would add 0.4 points.
