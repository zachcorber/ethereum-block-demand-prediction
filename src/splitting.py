from sklearn.preprocessing import StandardScaler
from src.config import TRAIN_RATIO, VAL_RATIO, TARGET_COL

def chronological_split(df, feature_cols):
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in dataframe.")

    n = len(df)
    if n == 0:
        raise ValueError("Cannot split an empty dataframe.")

    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise ValueError("One of the train/val/test splits is empty.")

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL].astype(int)

    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COL].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL].astype(int)

    return train_df, val_df, test_df, X_train, y_train, X_val, y_val, X_test, y_test

def scale_splits(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_val_scaled, X_test_scaled