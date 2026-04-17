import numpy as np
from src.config import BASE_FEATURES, ENGINEERED_FEATURE_CANDIDATES

def safe_div(a, b):
    b = b.replace(0, np.nan)
    return a / b

def engineer_features(df):
    df = df.copy()

    if "block_height" in df.columns:
        df = df.sort_values("block_height").reset_index(drop=True)

    # Domain features
    if {"gas_used", "gas_limit"}.issubset(df.columns):
        df["gas_used_ratio"] = safe_div(df["gas_used"], df["gas_limit"])

    if {"total_tx", "block_size"}.issubset(df.columns):
        df["tx_per_byte"] = safe_div(df["total_tx"], df["block_size"])

    if {"gas_used", "total_tx"}.issubset(df.columns):
        df["gas_per_tx"] = safe_div(df["gas_used"], df["total_tx"])

    # Log features
    for col in ["total_tx", "block_size", "gas_avg_price"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    # Lag features
    for col in ["total_tx", "gas_used", "gas_limit", "block_size"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # Rolling features
    if "total_tx" in df.columns:
        df["total_tx_roll5"] = df["total_tx"].rolling(5).mean()
        df["total_tx_roll20"] = df["total_tx"].rolling(20).mean()

    # Drop rows with NaNs caused by lag/rolling features
    df = df.dropna().reset_index(drop=True)

    base_features = [c for c in BASE_FEATURES if c in df.columns]
    engineered_features = base_features + [
        c for c in ENGINEERED_FEATURE_CANDIDATES if c in df.columns
    ]
    engineered_features = list(dict.fromkeys(engineered_features))

    return df, base_features, engineered_features