"""Score saved answers. LiveCodeBench executes generated Python: use an isolated machine."""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_verifier():
    checkout = Path(__file__).resolve().parent / "third_party" / "LiveCodeBench"
    if (checkout / "lcb_runner").is_dir():
        sys.path.insert(0, str(checkout))
    try:
        from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
        from lcb_runner.evaluation import codegen_metrics
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.utils.extraction_utils import extract_code
    except ModuleNotFoundError as exc:
        if exc.name == "lcb_runner":
            raise SystemExit(
                "LiveCodeBench is missing. From the project directory, run:\n"
                "  git clone https://github.com/LiveCodeBench/LiveCodeBench third_party/LiveCodeBench\n"
                "  git -C third_party/LiveCodeBench checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
            ) from exc
        raise SystemExit(
            f"Scoring dependency {exc.name!r} is missing from {sys.executable}. "
            "Install requirements.txt in the training/scoring environment."
        ) from exc
    return CodeGenerationProblem, codegen_metrics, LMStyle, extract_code


def aime_answer(text):
    # Score the final boxed integer, never an incidental number in the reasoning.
    text = str(text).split("</think>")[-1]
    boxes = re.findall(r"\\boxed\{\s*([0-9]{1,3})\s*\}", text)
    if boxes:
        return int(boxes[-1])
    match = re.fullmatch(r"\s*([0-9]{1,3})\s*", text)
    return int(match[1]) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--check-dependencies", action="store_true",
                        help="Check verifier imports without scoring answers")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=6)
    args = parser.parse_args()
    if min(args.workers, args.timeout) < 1:
        parser.error("workers and timeout must be positive")
    if not args.check_dependencies and args.input is None:
        parser.error("--input is required unless --check-dependencies is used")
    CodeGenerationProblem, codegen_metrics, LMStyle, extract_code = load_verifier()
    if args.check_dependencies:
        print("LiveCodeBench scoring dependencies are available.")
        return

    scores = {}
    for benchmark in ["aime24", "livecodebench_v5"]:
        rows = [json.loads(line) for line in (args.input / f"{benchmark}.jsonl").read_text().splitlines()]
        groups = defaultdict(list)
        for row in rows:
            groups[row["sample_id"]].append(row)
        if not groups or len({len(group) for group in groups.values()}) != 1:
            raise ValueError(f"{benchmark}: missing or incomplete samples")
        config = json.loads((args.input / "eval_config.json").read_text())
        expected_problems = min(config["limit"] or 10**9, 30 if benchmark == "aime24" else 167)
        if len(groups) != expected_problems or any(
            sorted(row["sample"] for row in group) != list(range(config["samples"]))
            for group in groups.values()
        ):
            raise ValueError(f"{benchmark}: generation incomplete; refusing a partial score")
        if benchmark == "aime24":
            correct = [aime_answer(row["completion"]) == int(row["answer"]) for row in rows]
            rate = sum(correct) / len(correct)
            details = [{"sample_id": row["sample_id"], "sample": row["sample"], "correct": ok}
                       for row, ok in zip(rows, correct)]
        else:
            samples, generations = [], []
            for group in groups.values():
                metadata = group[0]["metadata"]
                problem = CodeGenerationProblem(**(json.loads(metadata) if isinstance(metadata, str) else metadata))
                samples.append(problem.get_evaluation_sample())
                generations.append([extract_code(row["completion"], LMStyle.CodeQwenInstruct) for row in group])
            metrics, results, metadata = codegen_metrics(
                samples, generations, k_list=[1],
                num_process_evaluate=args.workers, timeout=args.timeout,
            )
            rate = float(metrics["pass@1"])
            details = {"sample_ids": list(groups), "results": results, "metadata": metadata}
        scores[benchmark] = {
            "pass@1": rate, "problems": len(groups), "samples_per_problem": len(rows) // len(groups),
            "hit_token_limit_fraction": sum(row["hit_token_limit"] for row in rows) / len(rows),
        }
        (args.input / f"{benchmark}_details.json").write_text(json.dumps(details, indent=2) + "\n")
    (args.input / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
