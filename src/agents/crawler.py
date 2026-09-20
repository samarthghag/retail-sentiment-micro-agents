"""Crawler micro-agent for querying and ingesting financial posts and microblogs."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class DataCrawlerAgent:
    """Agent responsible for streaming, filtering, and organizing financial social posts."""

    def __init__(self, processed_dir: Optional[Union[str, Path]] = None, config: Optional[Dict[str, Any]] = None):
        if processed_dir is None:
            raise ValueError("processed_dir path is required (e.g. CONFIG['processed_dir']).")
        self.processed_dir = Path(processed_dir)
        self.config = config or {}

        tweet_path = self.processed_dir / "tweet_df.parquet"
        if not tweet_path.exists():
            raise FileNotFoundError(f"Expected cached tweet data at {tweet_path}, but it does not exist.")
        self._tweet_df = pd.read_parquet(tweet_path)

    def fetch_records(self, ticker: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        """Fetch raw messages for a given ticker symbol within a timeframe.

        Args:
            ticker: Stock ticker symbol to filter by (e.g. "AAPL").
            start_time: ISO date string (YYYY-MM-DD), inclusive lower bound.
            end_time: ISO date string (YYYY-MM-DD), inclusive upper bound.

        Returns:
            List[Dict[str, Any]]: Matching records as plain dicts (ticker, date, text).
        """
        mask = (
            (self._tweet_df["ticker"] == ticker)
            & (self._tweet_df["date"] >= pd.to_datetime(start_time))
            & (self._tweet_df["date"] <= pd.to_datetime(end_time))
        )
        matched = self._tweet_df.loc[mask].sort_values("date")
        return matched.to_dict(orient="records")
