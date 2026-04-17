from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

# Kaggle dataset settings
KAGGLE_DATASET = "muhammedabdulazeem/ethereum-block-data"

PREFERRED_CSV_NAME = "eth.csv"

# Random seed
RANDOM_STATE = 42

# Chronological split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Outlier capping columns
CAP_COLS = [
    "block_reward",
    "gas_avg_price"
]

# Base supervised features
BASE_FEATURES = [
    "total_tx",
    "block_size",
    "gas_limit",
    "gas_used",
    "gas_avg_price",
    "block_time_in_sec",
    "hour"
]

# Engineered feature candidates
ENGINEERED_FEATURE_CANDIDATES = [
    "gas_used_ratio",
    "tx_per_byte",
    "gas_per_tx",
    "log_total_tx",
    "log_block_size",
    "log_gas_avg_price",
    "total_tx_lag1",
    "gas_used_lag1",
    "gas_limit_lag1",
    "block_size_lag1",
    "total_tx_roll5",
    "total_tx_roll20"
]

# Target settings
TARGET_SOURCE_COL = "total_tx"
TARGET_QUANTILE = 0.95
TARGET_COL = "target"
HIGH_DEMAND_COL = "high_demand"

# Model tuning grids
RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 5]
}

XGB_GRID = {
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05],
    "subsample": [0.7, 0.9],
    "colsample_bytree": [0.7, 0.9]
}

XGB_N_ESTIMATORS = 500

# Unsupervised settings
PCA_VARIANCE_THRESHOLD = 0.90
K_RANGE = range(2, 9)
SILHOUETTE_SAMPLE_SIZE = 10_000
PLOT_SAMPLE_SIZE = 20_000

#Evaluation Settings
DEFAULT_THRESHOLD = 0.5
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ROBUSTNESS_CHUNKS = 5