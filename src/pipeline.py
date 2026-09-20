"""LangGraph pipeline wiring the crawler, sentiment scorer, divergence detector, and aggregator agents together."""

from typing import Any, Dict, List, TypedDict

import pandas as pd
from langgraph.graph import StateGraph, END


class PipelineState(TypedDict, total=False):
    ticker: str
    start_time: str
    end_time: str
    tweets: List[Dict[str, Any]]
    sentiment_records: List[Dict[str, Any]]
    daily_sentiment: pd.Series
    daily_price: pd.Series
    divergence_df: pd.DataFrame
    signal_records: List[Dict[str, Any]]
    result_df: pd.DataFrame


def build_pipeline(crawler, scorer, detector, aggregator, price_df: pd.DataFrame):
    """Wire the four micro-agents into a compiled LangGraph pipeline.

    price_df must contain columns: ticker, date, adj_close (as loaded from
    the cached data/processed/price_df.parquet).
    """

    def crawl_node(state: PipelineState) -> Dict[str, Any]:
        tweets = crawler.fetch_records(state["ticker"], state["start_time"], state["end_time"])
        return {"tweets": tweets}

    def sentiment_node(state: PipelineState) -> Dict[str, Any]:
        tweets = state["tweets"]
        if not tweets:
            return {"sentiment_records": [], "daily_sentiment": pd.Series(dtype=float)}
        texts = [t["text"] for t in tweets]
        scores = scorer.score_batch(texts)
        for tweet, score in zip(tweets, scores):
            tweet["sentiment"] = score
        df = pd.DataFrame(tweets)
        df["strength"] = df["sentiment"].apply(lambda s: s.get("Positive", 0.0) - s.get("Negative", 0.0))
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        daily = df.groupby("date")["strength"].mean()
        return {"sentiment_records": tweets, "daily_sentiment": daily}

    def price_node(state: PipelineState) -> Dict[str, Any]:
        ticker_prices = price_df[price_df["ticker"] == state["ticker"]].copy()
        ticker_prices["date"] = pd.to_datetime(ticker_prices["date"]).dt.normalize()
        ticker_prices = ticker_prices.sort_values("date")
        ticker_prices["return"] = ticker_prices["adj_close"].pct_change()
        mask = (
            (ticker_prices["date"] >= pd.to_datetime(state["start_time"]))
            & (ticker_prices["date"] <= pd.to_datetime(state["end_time"]))
        )
        daily_price = ticker_prices.loc[mask].set_index("date")["return"].dropna()
        return {"daily_price": daily_price}

    def divergence_node(state: PipelineState) -> Dict[str, Any]:
        divergence_df = detector.detect_divergence(state["daily_sentiment"], state["daily_price"])
        return {"divergence_df": divergence_df}

    def aggregate_node(state: PipelineState) -> Dict[str, Any]:
        div_df = state["divergence_df"]
        daily_sentiment = state["daily_sentiment"]

        signal_records = []
        for date in daily_sentiment.index:
            strength = daily_sentiment.loc[date]
            row = div_df.loc[date] if date in div_df.index else None
            is_divergent = bool(row["is_divergent"]) if row is not None else False
            divergence_score = float(row["divergence_score"]) if row is not None else 0.0
            positive = max(strength, 0.0)
            negative = max(-strength, 0.0)
            signal_records.append(
                {
                    "ticker": state["ticker"],
                    "date": date,
                    "sentiment": {"Positive": positive, "Negative": negative},
                    "is_divergent": is_divergent,
                    "divergence_score": divergence_score,
                }
            )

        result_df = aggregator.aggregate_signals(signal_records)
        return {"signal_records": signal_records, "result_df": result_df}

    graph = StateGraph(PipelineState)
    graph.add_node("crawl", crawl_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("price", price_node)
    graph.add_node("divergence", divergence_node)
    graph.add_node("aggregate", aggregate_node)

    graph.set_entry_point("crawl")
    graph.add_edge("crawl", "sentiment")
    graph.add_edge("sentiment", "price")
    graph.add_edge("price", "divergence")
    graph.add_edge("divergence", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


def run_pipeline_for_tickers(
    compiled_graph,
    tickers: List[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """Run the compiled pipeline for each ticker and concatenate results into one ranked table."""
    all_results = []
    for ticker in tickers:
        final_state = compiled_graph.invoke(
            {"ticker": ticker, "start_time": start_time, "end_time": end_time}
        )
        result_df = final_state.get("result_df")
        if result_df is not None and not result_df.empty:
            all_results.append(result_df)

    if not all_results:
        return pd.DataFrame(columns=["ticker", "date", "sentiment_strength", "divergence_component", "final_score"])

    combined = pd.concat(all_results, ignore_index=True)
    combined = combined.sort_values("final_score", ascending=False).reset_index(drop=True)
    return combined
