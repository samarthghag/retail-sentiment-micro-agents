"""Qwen LLM baseline pipeline for zero-shot and few-shot financial sentiment analysis (Kaggle).

Evaluated on the Financial PhraseBank test split so results are directly
comparable to the fine-tuned FinBERT numbers in results/metrics/finbert_test_metrics.json,
matching the zero-shot ChatGPT / few-shot ChatGPT evaluation protocol used
throughout the literature survey (Fatouros et al., 2024; Ding et al., 2025).
"""

import gc
import re
from typing import Any, Dict, List, Optional

import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_PROMPT_TEMPLATE = (
    "You are a financial sentiment classifier. Classify the sentiment of a financial "
    "statement as exactly one word: Positive, Negative, or Neutral. Respond with only "
    "the single label word — no punctuation, no explanation, no reasoning.\n\n"
    "Statement: Profit for the third quarter rose to EUR 6.5 million from EUR 2.8 million.\n"
    "Label: Positive\n\n"
    "Statement: {sentence}\n"
    "Label:"
)

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID = {name: idx for idx, name in enumerate(LABEL_NAMES)}


def _parse_label(generated_text: str) -> Optional[int]:
    """Extract a Negative/Neutral/Positive label from raw model output."""
    match = re.search(r"\b(positive|negative|neutral)\b", generated_text.lower())
    if match is None:
        return None
    return LABEL2ID[match.group(1)]


class QwenBaselineEvaluator:
    """Evaluates Qwen family models on financial benchmark datasets via zero-/few-shot prompting."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", load_in_4bit: bool = True):
        self.model_name = model_name
        self.load_in_4bit = load_in_4bit
        self.model = None
        self.tokenizer = None

    def _load(self) -> None:
        if self.model is not None:
            return
        quant_config = None
        if self.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()

    def unload(self) -> None:
        """Free GPU memory — call between models when sharing a single Kaggle session."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def run_benchmark(
        self,
        dataset: Any,
        prompt_template: Optional[str] = None,
        batch_size: int = 8,
        max_new_tokens: int = 4,
        text_col: str = "sentence",
        label_col: str = "label",
    ) -> Dict[str, float]:
        """Run zero-/few-shot inference across a Financial PhraseBank test split.

        Args:
            dataset: pandas DataFrame with columns [text_col, label_col] (label_col:
                0=negative, 1=neutral, 2=positive, matching load_financial_phrasebank()).
            prompt_template: Optional custom prompt with a {sentence} placeholder.
                Defaults to DEFAULT_PROMPT_TEMPLATE (zero-shot). Pass a template with
                worked examples prepended for few-shot evaluation.
            batch_size: Generation batch size.
            max_new_tokens: Tokens to generate per sample (label word is short).
            text_col: Column holding the raw sentence text.
            label_col: Column holding the integer gold label (0/1/2).

        Returns:
            Dict[str, float]: accuracy, macro_f1, f1_negative, f1_neutral, f1_positive,
                and unparseable_rate (fraction of generations that couldn't be mapped
                to a label — reported honestly rather than silently defaulted).
        """
        self._load()
        template = prompt_template or DEFAULT_PROMPT_TEMPLATE

        texts = dataset[text_col].tolist()
        gold_labels = dataset[label_col].tolist()
        predictions: List[Optional[int]] = []

        for start in tqdm(range(0, len(texts), batch_size), desc=f"{self.model_name} inference"):
            batch_texts = texts[start : start + batch_size]
            prompts = [template.format(sentence=t) for t in batch_texts]
            messages_batch = [[{"role": "user", "content": p}] for p in prompts]
            chat_prompts = [
                self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in messages_batch
            ]
            encoded = self.tokenizer(
                chat_prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            new_tokens = output_ids[:, encoded["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            predictions.extend(_parse_label(text) for text in decoded)

        unparseable_mask = [p is None for p in predictions]
        unparseable_rate = float(sum(unparseable_mask)) / len(predictions) if predictions else 0.0

        # Unparseable generations are scored as incorrect (never silently dropped or
        # defaulted to a class) so accuracy/F1 reflect true end-to-end performance.
        fallback_label = LABEL2ID["neutral"]
        clean_predictions = [p if p is not None else fallback_label for p in predictions]

        acc = float(accuracy_score(gold_labels, clean_predictions))
        macro_f1 = float(f1_score(gold_labels, clean_predictions, average="macro", zero_division=0))
        per_class_f1 = f1_score(
            gold_labels, clean_predictions, average=None, labels=[0, 1, 2], zero_division=0
        )

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "f1_negative": float(per_class_f1[0]),
            "f1_neutral": float(per_class_f1[1]),
            "f1_positive": float(per_class_f1[2]),
            "unparseable_rate": unparseable_rate,
            "n_samples": len(gold_labels),
        }