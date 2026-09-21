# Retail Sentiment Micro-Agents

A **micro-agent pipeline** for financial sentiment analysis and sentiment–price divergence detection, built on [LangGraph](https://github.com/langchain-ai/langgraph) and fine-tuned [FinBERT](https://huggingface.co/yiyanghkust/finbert-tone). The system decomposes the classic "social-media sentiment → stock signal" workflow into four cooperating, independently testable agents and benchmarks them against monolithic LLM baselines (Qwen 2.5-7B, Mistral 7B).

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Micro-Agent Descriptions](#micro-agent-descriptions)
- [Datasets](#datasets)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [Step 1 — Data Preparation](#step-1--data-preparation)
  - [Step 2 — FinBERT Fine-Tuning](#step-2--finbert-fine-tuning)
  - [Step 3 — Run the Pipeline](#step-3--run-the-pipeline)
  - [Step 4 — Baselines (Kaggle)](#step-4--baselines-kaggle)
- [Configuration](#configuration)
- [License](#license)

---

## Overview

This repository accompanies a research paper investigating whether **decomposing** financial sentiment analysis into specialised micro-agents (crawl → score → detect divergence → aggregate) outperforms monolithic large-language-model prompting. The core hypothesis is that a fine-tuned, domain-specific model (FinBERT) orchestrated by a lightweight agent graph can match or exceed 7B-parameter instruction-tuned LLMs in both accuracy and cost-efficiency for retail-investor-oriented sentiment signals.

### Research Questions

1. Can a fine-tuned FinBERT model achieve competitive sentiment classification accuracy on financial text?
2. Does a multi-agent decomposition (crawl → score → divergence → aggregate) produce actionable divergence signals between social sentiment and market returns?
3. How do micro-agent pipelines compare to monolithic LLM baselines (Qwen 2.5-7B, Mistral 7B) in zero/few-shot settings?

---

## Architecture

The pipeline is orchestrated as a **LangGraph `StateGraph`**, with a typed state (`PipelineState`) flowing through five sequential nodes:

```
┌─────────┐     ┌────────────┐     ┌───────────┐     ┌─────────────┐     ┌────────────┐
│  Crawl  │ ──► │ Sentiment  │ ──► │   Price   │ ──► │ Divergence  │ ──► │ Aggregate  │
│  Agent  │     │  Scorer    │     │  Loader   │     │  Detector   │     │  Agent     │
└─────────┘     └────────────┘     └───────────┘     └─────────────┘     └────────────┘
    │                │                  │                  │                   │
 tweets         sentiment           daily              z-score            final_score
 (list)         records            returns            divergence          ranked table
                                  (Series)            flags (DF)            (DF)
```

Each ticker runs through the full graph independently. Results are concatenated and sorted by `final_score` descending.

---

## Micro-Agent Descriptions

| Agent | Class | Responsibility |
|---|---|---|
| **Crawler** | `DataCrawlerAgent` | Fetches cached StockNet tweets for a given ticker and date range from Parquet storage. |
| **Sentiment Scorer** | `SentimentScorerAgent` | Runs batch inference through a fine-tuned FinBERT checkpoint, returning per-class softmax probabilities (Positive / Neutral / Negative). |
| **Divergence Detector** | `DivergenceDetectorAgent` | Aligns daily sentiment strength with daily price returns, computes z-scores, and flags dates where the two signals disagree in direction **and** magnitude (configurable threshold, default 1.5σ). |
| **Signal Aggregator** | `SignalAggregatorAgent` | Combines sentiment strength and divergence magnitude into a weighted final score (default weights: 60% sentiment, 40% divergence). |

---

## Datasets

| Dataset | Purpose | Source |
|---|---|---|
| **StockNet** (price + tweets) | Historical OHLCV price data and aligned Twitter posts for 88 stock tickers (2014–2016). | [yumoxu/stocknet-dataset](https://github.com/yumoxu/stocknet-dataset) |
| **Financial PhraseBank** | 4,845 annotated financial sentences (positive / neutral / negative) used to fine-tune FinBERT. Agreement threshold: `Sentences_50Agree`. | [takala/financial_phrasebank (HuggingFace)](https://huggingface.co/datasets/takala/financial_phrasebank) |

Both datasets are **automatically downloaded** on first run if not already present locally.

---

## Key Results

### FinBERT Fine-Tuning (Financial PhraseBank, 70/15/15 split)

| Metric | Score |
|---|---|
| **Accuracy** | 85.28% |
| **Macro F1** | 83.30% |
| F1 — Negative | 81.77% |
| F1 — Neutral | 88.84% |
| F1 — Positive | 79.28% |

Training was run for **6 epochs** with a learning rate of `2e-5`, best checkpoint selected by validation macro-F1.

### Pipeline Smoke Test (AAPL + AMZN, Jan 1–15 2015)

The pipeline successfully identified divergence events (e.g., AMZN on 2015-01-12 with `divergence_component = 3.23` and `final_score = 1.45`) where social sentiment strongly disagreed with price direction, alongside many non-divergent days with scores near zero.

---

## Project Structure

```
retail-sentiment-micro-agents/
├── config/
│   └── global_config.py          # Hyperparameters, paths, seed management
├── src/
│   ├── agents/
│   │   ├── crawler.py            # DataCrawlerAgent
│   │   ├── sentiment_scorer.py   # SentimentScorerAgent (FinBERT inference)
│   │   ├── divergence_detector.py# DivergenceDetectorAgent (z-score divergence)
│   │   └── aggregator.py         # SignalAggregatorAgent (weighted scoring)
│   ├── baselines/
│   │   ├── qwen_baseline.py      # Qwen 2.5-7B zero/few-shot evaluator (WIP)
│   │   └── mistral_baseline.py   # Mistral 7B zero/few-shot evaluator (WIP)
│   ├── data_loading.py           # StockNet + Financial PhraseBank loaders
│   ├── finbert_finetune.py       # FinBERT fine-tuning & evaluation
│   └── pipeline.py               # LangGraph state-graph wiring
├── notebooks/
│   ├── part1_2_setup_data.ipynb  # Data download, EDA, and preprocessing
│   ├── part3_finbert_local.ipynb # FinBERT fine-tuning (local GPU)
│   ├── part4_pipeline_local.ipynb# End-to-end pipeline execution
│   └── part5_baselines_kaggle.ipynb # LLM baseline benchmarks (Kaggle GPU)
├── data/
│   ├── raw/                      # StockNet clone + Financial PhraseBank zip
│   └── processed/                # Parquet caches (price_df, tweet_df, train/val/test)
├── checkpoints/                  # Fine-tuned FinBERT weights
├── results/
│   ├── figures/                  # Training loss curves, etc.
│   ├── metrics/                  # JSON evaluation reports
│   └── pipeline_runs/            # CSV outputs from pipeline smoke tests
├── check_best_checkpoint.py      # Quick evaluation of best saved checkpoint
├── requirements.txt              # Pinned dependencies
├── .gitignore
├── LICENSE                       # GPL-3.0
└── README.md
```

---

## Setup & Installation

### Prerequisites

- **Python 3.10+**
- **CUDA 12.4+** (recommended for fine-tuning and inference; CPU fallback is supported)
- **Git** (for auto-cloning the StockNet dataset)

### Install

```bash
# Clone the repository
git clone https://github.com/samarthghag/retail-sentiment-micro-agents.git
cd retail-sentiment-micro-agents

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: If you do not have a CUDA GPU, PyTorch will fall back to CPU automatically. Fine-tuning will be significantly slower on CPU.

---

## Usage

The project is structured as a **five-part notebook workflow**, with each notebook corresponding to a research phase. All notebooks are in the `notebooks/` directory.

### Step 1 — Data Preparation

Open and run **`notebooks/part1_2_setup_data.ipynb`**. This notebook:
- Clones the StockNet dataset (price + tweets) into `data/raw/`
- Downloads and extracts the Financial PhraseBank dataset
- Performs exploratory data analysis (EDA)
- Splits the data into train/val/test partitions (70/15/15)
- Caches all DataFrames as Parquet files in `data/processed/`

### Step 2 — FinBERT Fine-Tuning

Open and run **`notebooks/part3_finbert_local.ipynb`**. This notebook:
- Loads the `yiyanghkust/finbert-tone` pre-trained checkpoint
- Fine-tunes on the Financial PhraseBank training split for 6 epochs
- Evaluates on the test split (accuracy, macro-F1, per-class F1)
- Saves the best checkpoint to `checkpoints/`

You can also quickly evaluate an existing checkpoint:

```bash
python check_best_checkpoint.py
```

### Step 3 — Run the Pipeline

Open and run **`notebooks/part4_pipeline_local.ipynb`**. This notebook:
- Instantiates all four micro-agents
- Wires them into the LangGraph pipeline
- Runs the pipeline for selected tickers over a date range
- Saves ranked results to `results/pipeline_runs/`

### Step 4 — Baselines (Kaggle)

Open and run **`notebooks/part5_baselines_kaggle.ipynb`** on a Kaggle GPU instance. This notebook benchmarks:
- **Qwen 2.5-7B-Instruct** — zero-shot and few-shot financial sentiment
- **Mistral 7B-Instruct-v0.3** — zero-shot and few-shot financial sentiment

> **Status**: Baseline evaluators are currently stubbed out (`NotImplementedError`). Full implementation is in progress.

---

## Configuration

All hyperparameters and paths are centralised in [`config/global_config.py`](config/global_config.py):

| Parameter | Default | Description |
|---|---|---|
| `seed` | `42` | Global random seed (Python, NumPy, PyTorch) |
| `finbert_base_checkpoint` | `yiyanghkust/finbert-tone` | Base model for FinBERT fine-tuning |
| `max_seq_len` | `128` | Maximum token sequence length |
| `train_batch_size` | `16` | Training batch size |
| `eval_batch_size` | `32` | Evaluation batch size |
| `num_train_epochs` | `6` | Fine-tuning epochs |
| `learning_rate` | `2e-5` | Peak learning rate (AdamW) |
| `weight_decay` | `0.01` | L2 regularization weight |
| `warmup_ratio` | `0.1` | Linear warmup fraction |
| `train_split` / `val_split` / `test_split` | `0.70` / `0.15` / `0.15` | Data partition ratios |
| `load_in_4bit` | `True` | Enable 4-bit quantization for baselines |

---

## License

This project is licensed under the **GNU General Public License v3.0** — see the [LICENSE](LICENSE) file for details.
