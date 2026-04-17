from pathlib import Path
import pandas as pd
import kagglehub
import os
from dotenv import load_dotenv

from src.config import RAW_DATA_DIR, KAGGLE_DATASET, PREFERRED_CSV_NAME
from src.utils import validate_required_columns

load_dotenv(override=False)

def download_kaggle_dataset(dataset_slug=KAGGLE_DATASET):
    """
    Downloads the Kaggle dataset using kagglehub and returns
    the local directory where files are stored.
    """
    if not dataset_slug or dataset_slug == "YOUR_USERNAME/YOUR_DATASET_NAME":
        raise ValueError(
            "KAGGLE_DATASET is not set in config.py. "
            "Please update it with your real Kaggle dataset slug."
        )

    try:
        download_path = kagglehub.dataset_download(dataset_slug)
        return Path(download_path)
    except Exception as e:
        has_env_creds = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
        creds_hint = (
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY as environment variables "
            "or create a .env file in the repo root with those keys."
            if not has_env_creds
            else "Kaggle credentials were found in the environment, but the download still failed."
        )
        raise RuntimeError(
            f"Failed to download Kaggle dataset '{dataset_slug}'. "
            f"Make sure your Kaggle credentials are configured correctly. {creds_hint}\nOriginal error: {e}"
        )

def find_csv_file(folder_path, preferred_name=None):
    """
    Finds the CSV file to use.
    If preferred_name is provided, it looks for that first.
    Otherwise it picks the largest CSV in the dataset folder.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder_path}")

    csv_files = list(folder_path.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in dataset folder: {folder_path}")

    if preferred_name:
        matches = [f for f in csv_files if f.name == preferred_name]
        if not matches:
            raise FileNotFoundError(
                f"Preferred CSV '{preferred_name}' not found in dataset."
            )
        return matches[0]

    # Fallback: choose the largest CSV file
    largest_csv = max(csv_files, key=lambda f: f.stat().st_size)
    return largest_csv

def load_data():
    """
    End-to-end Kaggle retrieval + CSV loading.
    """
    dataset_path = download_kaggle_dataset()
    csv_path = find_csv_file(dataset_path, preferred_name=PREFERRED_CSV_NAME)

    df = pd.read_csv(csv_path)

    required_cols = [
        "block_height",
        "created_ts",
        "total_tx",
        "block_size",
        "gas_limit",
        "gas_used",
        "gas_avg_price",
        "block_time_in_sec"
    ]
    validate_required_columns(df, required_cols)

    if df.empty:
        raise ValueError("Loaded dataset is empty.")

    print(f"Loaded dataset from: {csv_path}")
    print(f"Dataset shape: {df.shape}")

    return df, csv_path