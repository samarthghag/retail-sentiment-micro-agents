"""Data loading and preprocessing utilities for financial sentiment research.

Handles ingestion, normalization, and caching of:
- StockNet price histories (OHLCV, returns, movement labels)
- StockNet tweets (raw JSON/text, timestamps, ticker alignments)
- Financial PhraseBank (annotated financial sentences with sentiment labels)
"""

import glob
import json
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config.global_config import CONFIG


def _ensure_stocknet(data_dir: Union[str, Path]) -> Path:
    """Ensure StockNet dataset repository is available locally."""
    data_path = Path(data_dir)
    candidate = data_path / "stocknet-dataset"
    if candidate.exists() and (candidate / "price").exists() and (candidate / "tweet").exists():
        return candidate
    if (data_path / "price").exists() and (data_path / "tweet").exists():
        return data_path

    target_dir = candidate
    if not target_dir.exists():
        data_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-q", "https://github.com/yumoxu/stocknet-dataset.git", str(target_dir)],
            check=True,
        )
    return target_dir


def load_stocknet_prices(
    price_dir: Union[str, Path],
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and parse historical stock price files from the StockNet dataset.

    Args:
        price_dir: Path to directory containing raw StockNet price CSV files or root data dir.
        tickers: Optional list of stock ticker symbols to filter by. If None,
            all discovered tickers are loaded.
        start_date: Optional ISO date string (YYYY-MM-DD) for start of time range.
        end_date: Optional ISO date string (YYYY-MM-DD) for end of time range.

    Returns:
        pd.DataFrame: Cleaned dataframe containing price records across tickers.
    """
    stocknet_dir = _ensure_stocknet(price_dir)
    price_raw_dir = stocknet_dir / "price" / "raw"
    if not price_raw_dir.exists():
        if (Path(price_dir) / "raw").exists():
            price_raw_dir = Path(price_dir) / "raw"
        elif Path(price_dir).exists():
            price_raw_dir = Path(price_dir)

    ticker_files = sorted(glob.glob(os.path.join(str(price_raw_dir), "*.csv")))
    if not ticker_files:
        raise FileNotFoundError(f"No price CSV files found in {price_raw_dir}")

    price_dfs = []
    for fp in ticker_files:
        ticker = os.path.splitext(os.path.basename(fp))[0]
        if tickers is not None and ticker not in tickers:
            continue
        df = pd.read_csv(fp)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {"adj_close": "adj_close", "adjclose": "adj_close", "close": "close"}
        df = df.rename(columns=rename_map)
        df["ticker"] = ticker
        price_dfs.append(df)

    if not price_dfs:
        return pd.DataFrame(columns=["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"])

    price_df = pd.concat(price_dfs, ignore_index=True)
    price_df["date"] = pd.to_datetime(price_df["date"])

    if start_date is not None:
        price_df = price_df[price_df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        price_df = price_df[price_df["date"] <= pd.to_datetime(end_date)]

    price_df = price_df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    return price_df


def load_stocknet_tweets(
    tweet_dir: Union[str, Path],
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and extract StockNet tweet records across target tickers.

    Args:
        tweet_dir: Path to directory containing tweet subfolders or root data dir.
        tickers: Optional list of stock ticker symbols to include.
        start_date: Optional ISO date string filter for tweets.
        end_date: Optional ISO date string filter for tweets.

    Returns:
        pd.DataFrame: Normalized dataframe containing tweet records.
    """
    stocknet_dir = _ensure_stocknet(tweet_dir)
    tweet_raw_dir = stocknet_dir / "tweet" / "raw"
    if not tweet_raw_dir.exists():
        if (Path(tweet_dir) / "raw").exists():
            tweet_raw_dir = Path(tweet_dir) / "raw"
        elif Path(tweet_dir).exists():
            tweet_raw_dir = Path(tweet_dir)

    if not os.path.exists(tweet_raw_dir):
        raise FileNotFoundError(f"No tweet raw directory found in {tweet_raw_dir}")

    tickers_with_tweets = sorted(os.listdir(str(tweet_raw_dir)))
    tweet_records = []
    for ticker in tqdm(tickers_with_tweets, desc="Loading tweets"):
        if tickers is not None and ticker not in tickers:
            continue
        ticker_dir = os.path.join(str(tweet_raw_dir), ticker)
        if not os.path.isdir(ticker_dir):
            continue
        for date_str in sorted(os.listdir(ticker_dir)):
            if start_date is not None and date_str < str(start_date):
                continue
            if end_date is not None and date_str > str(end_date):
                continue
            fpath = os.path.join(ticker_dir, date_str)
            if not os.path.isfile(fpath):
                continue
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", None)
                        if text is None:
                            continue
                        tweet_records.append({"ticker": ticker, "date": date_str, "text": text})
                    except json.JSONDecodeError:
                        continue

    if not tweet_records:
        return pd.DataFrame(columns=["ticker", "date", "text"])

    tweet_df = pd.DataFrame(tweet_records)
    tweet_df["date"] = pd.to_datetime(tweet_df["date"])
    if start_date is not None:
        tweet_df = tweet_df[tweet_df["date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        tweet_df = tweet_df[tweet_df["date"] <= pd.to_datetime(end_date)]

    tweet_df = tweet_df.sort_values(by=["ticker", "date"]).reset_index(drop=True)
    return tweet_df


def load_financial_phrasebank(
    data_path: Union[str, Path],
    split_ratio: float = 0.8,
    agreement_threshold: str = "Sentences_50Agree",
) -> Dict[str, pd.DataFrame]:
    """Load and prepare the Financial PhraseBank dataset for sentiment classification.

    Args:
        data_path: Path to the raw data directory or Financial PhraseBank folder.
        split_ratio: Fraction of data assigned to training partition if CONFIG splits are absent.
        agreement_threshold: Agreement subset to load (defaults to 'Sentences_50Agree').

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with 'train', 'val', and 'test' partitions
            containing columns: ['sentence', 'label_name', 'label']
    """
    fpb_dir = Path(data_path)
    fpb_extract_dir = fpb_dir / "FinancialPhraseBank-v1.0"
    fpb_zip_path = fpb_dir / "FinancialPhraseBank-v1.0.zip"
    fpb_zip_url = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"

    if not fpb_extract_dir.exists():
        fpb_dir.mkdir(parents=True, exist_ok=True)
        if not fpb_zip_path.exists():
            r = requests.get(fpb_zip_url, stream=True)
            r.raise_for_status()
            with open(fpb_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        with zipfile.ZipFile(fpb_zip_path, "r") as zf:
            zf.extractall(fpb_dir)

    target_suffix = agreement_threshold if agreement_threshold.endswith(".txt") else f"{agreement_threshold}.txt"
    candidate_files = [p for p in fpb_dir.rglob("*.txt") if target_suffix.lower() in p.name.lower()]
    if not candidate_files:
        raise FileNotFoundError(f"Could not find agreement file matching {agreement_threshold} in {fpb_dir}")
    fpb_file = candidate_files[0]

    records = []
    with open(fpb_file, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit("@", 1)
            if len(parts) != 2:
                continue
            sentence, label_name = parts[0].strip(), parts[1].strip()
            records.append({"sentence": sentence, "label_name": label_name})

    fpb_df = pd.DataFrame(records)
    label_map = {"negative": 0, "neutral": 1, "positive": 2}
    fpb_df["label"] = fpb_df["label_name"].map(label_map)
    assert fpb_df["label"].isnull().sum() == 0, "Detected unmapped sentiment labels in Financial PhraseBank"

    train_ratio = CONFIG.get("train_split", 0.70)
    val_ratio = CONFIG.get("val_split", 0.15)
    test_ratio = CONFIG.get("test_split", 0.15)
    seed = CONFIG.get("seed", 42)

    train_df, temp_df = train_test_split(
        fpb_df,
        train_size=train_ratio,
        stratify=fpb_df["label"],
        random_state=seed,
    )
    val_relative = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_relative,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def cache_dataframes_to_parquet(
    data_frames: Dict[str, pd.DataFrame],
    output_dir: Union[str, Path],
    compression: str = "snappy",
) -> Dict[str, Path]:
    """Serialize and cache processed DataFrames to columnar Parquet format.

    Args:
        data_frames: Dictionary mapping cache names (e.g., 'price_df', 'train')
            to their respective pandas DataFrames.
        output_dir: Destination folder path for storing parquet files.
        compression: Compression codec to use (defaults to 'snappy').

    Returns:
        Dict[str, Path]: Mapping of dataset keys to the file paths written on disk.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: Dict[str, Path] = {}
    for name, df in data_frames.items():
        path = output_path / f"{name}.parquet"
        df.to_parquet(path, index=False, compression=compression)
        saved_paths[name] = path
    return saved_paths
