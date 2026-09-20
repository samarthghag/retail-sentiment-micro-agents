"""Divergence detector micro-agent for spotting sentiment vs price dislocations."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class DivergenceDetectorAgent:
    """Agent that identifies anomalies and divergence between market returns and sentiment."""

    def __init__(self, threshold: float = 1.5):
        self.threshold = threshold

    def detect_divergence(
        self,
        sentiment_series: pd.Series,
        price_series: pd.Series,
    ) -> pd.DataFrame:
        """Calculate divergence flags and magnitude between sentiment movement and price trajectory.

        Combines two checks:
          1. Direction mismatch: sentiment and price moved in opposite directions.
          2. Magnitude gap: the z-scored difference between how unusual each move
             is, relative to its own series' history.

        A row is flagged as divergent only when BOTH hold: the two signals
        disagree in direction, AND that disagreement is statistically
        significant (exceeds `self.threshold` in z-score terms) rather than
        noise.

        Args:
            sentiment_series: Time-indexed sentiment values (e.g. daily aggregate score).
            price_series: Time-indexed price return values (e.g. daily pct_change).

        Returns:
            pd.DataFrame: One row per aligned timestamp, containing the raw values,
                z-scores, a direction_mismatch flag, a divergence_score, and the
                final is_divergent boolean flag.
        """
        aligned = pd.concat(
            [sentiment_series.rename("sentiment"), price_series.rename("price")],
            axis=1,
            join="inner",
        ).dropna()

        if aligned.empty:
            return pd.DataFrame(
                columns=["sentiment", "price", "z_sentiment", "z_price", "direction_mismatch", "divergence_score", "is_divergent"]
            )

        sentiment_std = aligned["sentiment"].std()
        price_std = aligned["price"].std()

        aligned["z_sentiment"] = 0.0 if sentiment_std == 0 else (aligned["sentiment"] - aligned["sentiment"].mean()) / sentiment_std
        aligned["z_price"] = 0.0 if price_std == 0 else (aligned["price"] - aligned["price"].mean()) / price_std

        aligned["direction_mismatch"] = np.sign(aligned["sentiment"]) != np.sign(aligned["price"])
        aligned["divergence_score"] = (aligned["z_sentiment"] - aligned["z_price"]).abs()
        aligned["is_divergent"] = aligned["direction_mismatch"] & (aligned["divergence_score"] > self.threshold)

        return aligned
