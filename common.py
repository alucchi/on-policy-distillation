"""Shared checkpoints and the released Open-MOPD prompts."""

from datasets import load_dataset

PREFIX = "BytedTsinghua-SIA/Open-MOPD-SmolLM3-3B-"
MODELS = {"student": PREFIX + "MixSFT", "math": PREFIX + "RL-Math", "code": PREFIX + "RL-Code"}
DATASET = "BytedTsinghua-SIA/Open-MOPD-Data"


def load_prompts(config):
    return load_dataset(DATASET, config, split="train")
