"""Global configuration settings for the retail sentiment micro-agents pipeline."""

import random
from pathlib import Path
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG = {
    "seed": 42,
    "finbert_base_checkpoint": "yiyanghkust/finbert-tone",
    "monolithic_baseline_a": "Qwen/Qwen2.5-7B-Instruct",
    "monolithic_baseline_b": "mistralai/Mistral-7B-Instruct-v0.3",
    "fpb_config": "sentences_50agree",
    "max_seq_len": 128,
    "train_batch_size": 16,
    "eval_batch_size": 32,
    "num_train_epochs": 6,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "train_split": 0.70,
    "val_split": 0.15,
    "test_split": 0.15,
    "load_in_4bit": True,
    "data_dir": str(BASE_DIR / "data" / "raw"),
    "processed_dir": str(BASE_DIR / "data" / "processed"),
    "checkpoint_dir": str(BASE_DIR / "checkpoints"),
}


def set_seed(seed: int = CONFIG["seed"]) -> None:
    """Set random seed across random, numpy, and torch for reproducible execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CONFIG["seed"])
