"""Sentiment scorer micro-agent using fine-tuned FinBERT models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentScorerAgent:
    """Agent responsible for inferring sentiment polarities and confidence distributions."""

    def __init__(self, model_checkpoint: Optional[Union[str, Path]] = None, device: Optional[str] = None):
        if model_checkpoint is None:
            raise ValueError("model_checkpoint path is required (e.g. CONFIG['checkpoint_dir']).")
        self.model_checkpoint = str(model_checkpoint)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_checkpoint)
        self.model.to(self.device)
        self.model.eval()

        self.id2label: Dict[int, str] = self.model.config.id2label

    def score_batch(self, texts: List[str], batch_size: int = 32, max_seq_len: int = 128) -> List[Dict[str, float]]:
        """Compute sentiment scores for a collection of financial texts.

        Returns a list of dicts, one per input text, each containing:
            - "label": predicted class name (from the model's id2label mapping)
            - one key per class with its softmax probability
        """
        if not texts:
            return []

        results: List[Dict[str, float]] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = F.softmax(logits, dim=-1)

            for row in probs.cpu():
                class_probs = {self.id2label[i]: float(row[i]) for i in range(len(row))}
                predicted_label = self.id2label[int(row.argmax())]
                results.append({"label": predicted_label, **class_probs})

        return results
