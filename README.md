# Reasoning-Augmented Time-Series Forecasting (ARIMA + LLM)

This repository contains the official code for the **Agenthon-2026 Track 2** competition submission, implementing a **reasoning-augmented forecasting framework** that combines a classical statistical model (ARIMA) with a Large Language Model (LLM) to improve probabilistic forecasting of financial time series.

## Overview

Traditional numerical models (e.g., Random Walk, ARIMA) are often blind to qualitative information such as central bank statements and news headlines. This project bridges that gap by integrating **multi-dimensional text signals** extracted via an LLM into the ARIMA predictive distribution. The LLM outputs three key signals:

- **Sentiment Score** (−1 to +1): overall market sentiment.
- **Tail Risk Flag** (True/False): presence of extreme risk events.
- **Volatility Adjustment Factor** (1.0 to 2.0): expected amplification of market volatility.

These signals dynamically adjust the mean and variance of the ARIMA forecast, providing a more robust prediction of tail risks and extreme market events.

## Key Features

- **ARIMA(1,1,1) Base Model** with a fallback to a random-walk model when data is insufficient.
- **LLM Integration** via a customizable prompt that forces structured JSON output.
- **Strict Data Leakage Prevention**: uses the `data_cutoff` field from `card.toml` to enforce the as-of date and filters all data with `df["date"] <= pd.to_datetime(asof)`.
- **Full Compliance**: All experiments pass the official g0–g3 admissibility gates.

## Installation

### Prerequisites

- Python 3.13 or higher (tested with Conda environment `qfbench`)
- Docker (for building the submission image)

### Setup

```bash
# Clone the repository (or use your local copy)
git clone https://github.com/deldjo/reasoning-augmented-forecasting.git
cd reasoning-augmented-forecasting

# Create and activate a Conda environment (recommended)
conda create -n qfbench python=3.13 -y
conda activate qfbench

# Install dependencies
pip install --upgrade pip
pip install numpy pandas pyarrow tomli requests statsmodels scipy