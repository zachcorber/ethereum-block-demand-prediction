from pathlib import Path
from src.config import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR, RAW_DATA_DIR

def ensure_directories():
    for path in [RAW_DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        Path(path).mkdir(parents=True, exist_ok=True)

def validate_required_columns(df, required_cols):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")