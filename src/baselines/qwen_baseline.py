"""Qwen LLM baseline pipeline for zero-shot and few-shot financial sentiment analysis (Kaggle)."""

from typing import Any, Dict, List, Optional


class QwenBaselineEvaluator:
    """Evaluates Qwen family models on financial benchmark datasets."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
        self.model_name = model_name

    def run_benchmark(self, dataset: Any, prompt_template: Optional[str] = None) -> Dict[str, float]:
        """Run inference across test dataset and calculate baseline performance metrics."""
        raise NotImplementedError("QwenBaselineEvaluator will be implemented in Part 5.")
