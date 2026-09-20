"""FinBERT fine-tuning and evaluation module for financial domain adaptation.

Provides functions to:
- Instantiate pre-trained FinBERT weights and classification heads
- Tokenize and encode financial text using HuggingFace Tokenizers
- Execute fine-tuning routines with learning rate scheduling and early stopping
- Evaluate sentiment classification accuracy and macro F1 scores
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score


class FinancialDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized financial texts and optional labels."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[List[int]] = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])


def load_finbert_model(
    checkpoint_name: str = "yiyanghkust/finbert-tone",
    num_labels: int = 3,
    device: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load pre-trained FinBERT model and its corresponding AutoTokenizer.

    Args:
        checkpoint_name: HuggingFace model identifier or local checkpoint path.
        num_labels: Number of target sentiment classes (default: 3).
        device: Target compute device ('cuda', 'cpu', or auto-detected if None).

    Returns:
        Tuple[Any, Any]: (model, tokenizer) ready for tokenization and fine-tuning.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, use_fast=False)

    id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer


def tokenize_dataset(
    texts: list,
    tokenizer: Any,
    labels: Optional[list] = None,
    max_length: int = 128,
) -> Dataset:
    """Tokenize and prepare input encodings for transformer training/evaluation.

    Args:
        texts: List of raw input sentences.
        tokenizer: Pre-trained HuggingFace tokenizer instance.
        labels: Optional list of numeric target sentiment labels.
        max_length: Maximum token sequence length (defaults to 128).

    Returns:
        Dataset: PyTorch Dataset containing token tensors and optional labels.
    """
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return FinancialDataset(encodings, labels)


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute evaluation metrics for HuggingFace Trainer."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": macro_f1}


def train_finbert(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    val_dataset: Any,
    output_dir: Union[str, Path],
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute fine-tuning loop for FinBERT on financial domain data.

    Args:
        model: HuggingFace model instance for sequence classification.
        tokenizer: HuggingFace tokenizer paired with `model`, saved alongside
            each checkpoint so `from_pretrained(checkpoint_dir)` is self-contained.
        train_dataset: PyTorch Dataset containing training samples.
        val_dataset: PyTorch Dataset containing validation samples.
        output_dir: File path where model weights and training logs are saved.
        epochs: Number of training epochs (default: 3).
        batch_size: Batch size for training and validation loaders (default: 16).
        learning_rate: Peak learning rate for AdamW optimizer (default: 2e-5).
        seed: Random seed for reproducibility.

    Returns:
        Dict[str, Any]: Training history summary containing per-epoch loss and validation metrics.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(out_path),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_strategy": "epoch",
        "save_total_limit": 2,
    }

    try:
        training_args = TrainingArguments(eval_strategy="epoch", **training_args_kwargs)
    except TypeError:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,  # ensures tokenizer is written into each epoch checkpoint dir
    )

    train_result = trainer.train()
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))  # explicit belt-and-suspenders save at the final output_dir

    history: Dict[str, Any] = {
        "log_history": trainer.state.log_history,
        "train_loss": [e["loss"] for e in trainer.state.log_history if "loss" in e],
        "eval_loss": [e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e],
        "eval_accuracy": [e["eval_accuracy"] for e in trainer.state.log_history if "eval_accuracy" in e],
        "eval_macro_f1": [e["eval_macro_f1"] for e in trainer.state.log_history if "eval_macro_f1" in e],
        "train_runtime": train_result.metrics.get("train_runtime", 0.0),
    }

    return history


def evaluate_finbert(
    model: Any,
    test_dataset: Any,
    batch_size: int = 32,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Compute performance metrics for a trained FinBERT model on evaluation data.

    Computes accuracy, macro-F1, and per-class F1 scores.

    Args:
        model: Trained FinBERT model instance.
        test_dataset: PyTorch Dataset containing evaluation samples.
        batch_size: Evaluation batch size (default: 32).
        device: Target compute device for evaluation.

    Returns:
        Dict[str, float]: Evaluation metrics report mapping metric names to score values.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.get("labels")
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            if labels is not None:
                all_labels.extend(labels.cpu().numpy().tolist())

    metrics: Dict[str, float] = {}
    if all_labels:
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        class_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        label_names = ["negative", "neutral", "positive"]
        for idx, f1_val in enumerate(class_f1s):
            name = label_names[idx] if idx < len(label_names) else f"class_{idx}"
            metrics[f"f1_{name}"] = float(f1_val)

    return metrics
