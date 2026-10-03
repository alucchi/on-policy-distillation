"""Merge the math and code teachers by task arithmetic (Ilharco et al., ICLR 2023).

Each teacher d was fine-tuned from the same MixSFT weights theta_0, giving a task vector
tau_d = theta_d - theta_0. The merged model is theta_0 + lambda * sum_d tau_d, with one
scaling coefficient shared by all task vectors.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from common import MODELS

# Tokenizer, chat template and generation settings are copied from the base model.
BASE_FILES = ["config.json", "generation_config.json", "chat_template.jinja",
              "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]


def load_weights(name):
    path = Path(name) if Path(name).is_dir() else Path(snapshot_download(name))
    weights = {}
    for file in sorted(path.glob("*.safetensors")):
        weights.update(load_file(file))
    return path, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=MODELS["student"])
    parser.add_argument("--models", nargs="+", default=[MODELS["math"], MODELS["code"]])
    parser.add_argument("--lam", type=float, required=True, help="Scaling coefficient lambda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                        help="Saved precision; BF16 rounds away most of a fractional lambda update")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    base_path, base = load_weights(args.base)
    tasks = [load_weights(name)[1] for name in args.models]
    for name, weights in zip(args.models, tasks):
        if weights.keys() != base.keys() or any(weights[k].shape != base[k].shape for k in base):
            raise ValueError(f"{name} does not share the base model's parameters")

    dtype = getattr(torch, args.dtype)
    merged, stats = {}, {"lambda": args.lam, "models": args.models, "base": args.base, "dtype": args.dtype}
    norms, dots, update_norm, rounding = [0.0] * len(tasks), 0.0, 0.0, 0.0
    for key, theta_0 in base.items():
        # Differences of BF16 weights are exact in FP64 (FP32 loses a few tiny entries).
        vectors = [weights[key].double() - theta_0.double() for weights in tasks]
        update = args.lam * sum(vectors)
        exact = theta_0.double() + update
        merged[key] = exact.to(dtype)
        norms = [total + vector.pow(2).sum().item() for total, vector in zip(norms, vectors)]
        if len(vectors) == 2:
            dots += (vectors[0] * vectors[1]).sum().item()
        update_norm += update.pow(2).sum().item()
        rounding += (merged[key].double() - exact).pow(2).sum().item()
    stats["task_vector_norms"] = [norm ** 0.5 for norm in norms]
    if len(tasks) == 2:
        stats["task_vector_cosine"] = dots / (norms[0] * norms[1]) ** 0.5
    stats["update_norm"] = update_norm ** 0.5
    # Part of the update lost by rounding to the saved precision.
    stats["rounding_error_norm"] = rounding ** 0.5

    output.mkdir(parents=True, exist_ok=True)
    save_file(merged, output / "model.safetensors", metadata={"format": "pt"})
    for file in BASE_FILES:
        if (base_path / file).exists():
            shutil.copy(base_path / file, output / file)
    config = json.loads((output / "config.json").read_text())
    config["dtype"] = args.dtype
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "merge_config.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
