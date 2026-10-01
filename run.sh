#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/third_party/LiveCodeBench${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"
EVAL_PYTHON="${EVAL_PYTHON:-$PYTHON}"
OUTPUT="${OUTPUT:-outputs}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-mopd-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
# Fail before generating answers if the scoring environment is incomplete.
"$PYTHON" score.py --check-dependencies
generate_if_missing() {
    local model="$1"
    local destination="$2"
    if [[ -f "$destination/aime24.jsonl" && -f "$destination/livecodebench_v5.jsonl" && -f "$destination/eval_config.json" ]]; then
        echo "Reusing generation files in $destination; skipping generation."
    elif [[ -e "$destination/aime24.jsonl" || -e "$destination/livecodebench_v5.jsonl" || -e "$destination/eval_config.json" ]]; then
        echo "Incomplete generation files in $destination. Use a new OUTPUT directory or move the incomplete files before retrying." >&2
        return 1
    else
        "$EVAL_PYTHON" generate.py --model "$model" --output "$destination" \
            --limit "${LIMIT:-0}" --samples "${SAMPLES:-1}" \
            --max-new-tokens "${EVAL_TOKENS:-32768}"
    fi
}

# Set LIMIT=5 for a smoke run; 0 uses all 30 AIME24 and 167 LCB v5 problems.
# Baselines generate in parallel, one GPU queue each (EVAL_GPUS, round-robin when there
# are fewer GPUs than models). Each model's generation output goes to $OUTPUT/logs/.
BASELINES=(MixSFT RL-Math RL-Code)
IFS=, read -ra EVAL_GPU_LIST <<< "${EVAL_GPUS:-0,1,2}"
mkdir -p "$OUTPUT/logs"
pids=()
for index in "${!EVAL_GPU_LIST[@]}"; do
    (
        for ((model = index; model < ${#BASELINES[@]}; model += ${#EVAL_GPU_LIST[@]})); do
            name="${BASELINES[model]}"
            echo "Generating $name on GPU ${EVAL_GPU_LIST[index]} (log: $OUTPUT/logs/eval_$name.log)"
            CUDA_VISIBLE_DEVICES="${EVAL_GPU_LIST[index]}" generate_if_missing \
                "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-$name" "$OUTPUT/$name" \
                > "$OUTPUT/logs/eval_$name.log" 2>&1 \
                || { echo "Generation failed for $name; see $OUTPUT/logs/eval_$name.log" >&2; exit 1; }
        done
    ) &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
done
(( failed == 0 )) || exit 1
for name in "${BASELINES[@]}"; do
    "$PYTHON" score.py --input "$OUTPUT/$name"
done
"$PYTHON" train.py --output "$OUTPUT/student" --steps "${STEPS:-100}" \
    --max-new-tokens "${TRAIN_TOKENS:-4096}" --lr "${LR:-2e-6}" \
    --student-device "${STUDENT_DEVICE:-cuda:1}" --teacher-device "${TEACHER_DEVICE:-cuda:1}"
generate_if_missing "$OUTPUT/student" "$OUTPUT/distilled"
"$PYTHON" score.py --input "$OUTPUT/distilled"
"$PYTHON" - "$OUTPUT" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
print(f"{'Model':<14} {'AIME24':>10} {'LCB v5':>10}")
for name in ['MixSFT', 'RL-Math', 'RL-Code', 'distilled']:
    scores = json.loads((root / name / 'scores.json').read_text())
    print(f"{name:<14} {scores['aime24']['pass@1']:>10.1%} {scores['livecodebench_v5']['pass@1']:>10.1%}")
PY
