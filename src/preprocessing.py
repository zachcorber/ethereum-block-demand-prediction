import numpy as np
import pandas as pd
from src.config import CAP_COLS, TARGET_SOURCE_COL, TARGET_QUANTILE, TARGET_COL, HIGH_DEMAND_COL

def cap_outliers_iqr(df, cols):
    df = df.copy()

    for col in cols:
        if col not in df.columns:
            continue

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df[col] = df[col].clip(lower=lower, upper=upper)

    return df

def preprocess_data(df):
    df = df.copy()

    # Drop columns from notebook
    for col in ["miner_icon_url", "id"]:
        if col in df.columns:
            df = df.drop(columns=col)

    # Sort chronologically
    if "block_height" not in df.columns:
        raise ValueError("block_height column is required for chronological sorting.")
    df = df.sort_values("block_height").reset_index(drop=True)

    # Convert timestamp
    if "created_ts" not in df.columns:
        raise ValueError("created_ts column is required.")
    df["created_ts"] = pd.to_datetime(df["created_ts"], unit="s", errors="coerce")

    if df["created_ts"].isna().any():
        raise ValueError("Some created_ts values could not be converted to datetime.")

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Cap outliers
    df = cap_outliers_iqr(df, CAP_COLS)

    # Time feature from notebook
    df["hour"] = df["created_ts"].dt.hour

    # Binary target setup from notebook
    threshold = df[TARGET_SOURCE_COL].quantile(TARGET_QUANTILE)
    df[HIGH_DEMAND_COL] = (df[TARGET_SOURCE_COL] >= threshold).astype(int)

    # Predict next block demand
    df[TARGET_COL] = df[HIGH_DEMAND_COL].shift(-1)

    # Drop final row with missing future target
    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    return df, threshold