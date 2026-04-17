# Ethereum Block Demand Prediction

This repository contains an end-to-end machine learning pipeline that analyzes Ethereum block-level data and predicts whether the **next block** will fall into a **high-demand** regime (i.e., unusually high transaction activity). It also includes a Streamlit dashboard for exploring the generated evaluation tables and figures.

Author: Zachary Corber

## What this project does

- Downloads the configured Ethereum block dataset from Kaggle.
- Builds a supervised learning target: whether the **next** block is high-demand.
- Trains and compares multiple models (baseline + advanced tree-based models).
- Produces evaluation artifacts (metrics tables, curves, threshold analysis, temporal robustness).
- Runs an unsupervised workflow (PCA + KMeans) to discover activity regimes.
- Saves all artifacts under `outputs/` and visualizes them in a Streamlit dashboard.

## Outputs

After running the pipeline, you should see files created under:

- `outputs/tables/` (CSV summaries such as validation/test metrics and robustness)
- `outputs/figures/` (model curves, feature importance, interpretation plots)

## Download & setup

### Prerequisites

- Python 3.11+ recommended
- A Kaggle account with an API token (for programmatic dataset download)

### 1) Clone the repository

```bash
git clone <YOUR_REPO_URL>
cd ethereum-block-demand-prediction
```

### 2) Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Kaggle credentials (.env)

The pipeline downloads the dataset via Kaggle. Configure your Kaggle credentials using environment variables.

1) Create a file named `.env` in the repo root (this repo’s `.gitignore` ignores it).

2) Add:

```text
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

To generate an API token/key on Kaggle: go to your Kaggle account settings and create a new API token.

## Run the pipeline (recommended)

This runs the full workflow (download → preprocess → train → evaluate → interpretation → clustering) and writes results to `outputs/`.

```bash
python run_pipeline.py
```

Notes:

- The dataset is large; the first run can take significant time and disk.
- The pipeline creates required folders automatically.

## Run the dashboard

After the pipeline finishes (so `outputs/` exists), launch the Streamlit app:

```bash
streamlit run app.py
```

If the dashboard reports missing tables/figures, rerun `python run_pipeline.py`.