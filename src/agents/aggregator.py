"""Aggregator micro-agent synthesizing multi-source signals and agent outputs."""

from typing import Any, Dict, List, Optional

import pandas as pd


class SignalAggregatorAgent:
    """Agent that aggregates micro-agent signals into unified trade/sentiment decisions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {"sentiment": 0.6, "divergence": 0.4}

    def aggregate_signals(self, signal_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Combine heterogeneous agent outputs into final score rankings.

        Each record in signal_records is expected to optionally contain:
            - "ticker", "date": identifying fields, carried through unchanged.
            - "sentiment": a dict with "Positive"/"Negative" probability keys,
              as returned by SentimentScorerAgent.score_batch().
            - "is_divergent", "divergence_score": as returned by
              DivergenceDetectorAgent.detect_divergence().

        Missing fields default to a neutral contribution (0.0) rather than
        raising an error, so partial signal sets can still be aggregated.

        Returns:
            pd.DataFrame: One row per input record, with sentiment_strength,
                divergence_component, and final_score columns, sorted by
                final_score descending.
        """
        rows = []
        for record in signal_records:
            sentiment = record.get("sentiment", {})
            sentiment_strength = sentiment.get("Positive", 0.0) - sentiment.get("Negative", 0.0)

            is_divergent = record.get("is_divergent", False)
            divergence_score = record.get("divergence_score", 0.0)
            divergence_component = divergence_score if is_divergent else 0.0

            final_score = (
                self.weights["sentiment"] * sentiment_strength
                + self.weights["divergence"] * divergence_component
            )

            rows.append(
                {
                    "ticker": record.get("ticker"),
                    "date": record.get("date"),
                    "sentiment_strength": sentiment_strength,
                    "divergence_component": divergence_component,
                    "final_score": final_score,
                }
            )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
        return result
