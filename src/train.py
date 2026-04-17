import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from tqdm import tqdm

from src.config import RANDOM_STATE, RF_GRID, XGB_GRID, XGB_N_ESTIMATORS
from src.evaluation import eval_binary


def get_scale_pos_weight(y):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return neg / max(pos, 1)


def train_baseline_logistic(X_train_scaled, y_train, X_val_scaled, y_val):
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=1000,
        class_weight="balanced",
        n_jobs=None
    )

    result = eval_binary(
        "Logistic Regression (baseline)",
        model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val
    )

    return model, result


def tune_random_forest(X_train_scaled, y_train, X_val_scaled, y_val):
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(
        len(X_train_scaled),
        size=min(200000, len(X_train_scaled)),
        replace=False
    )
    X_tune = X_train_scaled[idx]
    y_tune = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

    best_auc = -1
    best_params = None

    for params in tqdm(list(ParameterGrid(RF_GRID)), desc="Tuning Random Forest"):
        model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            **params
        )
        model.fit(X_tune, y_tune)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, probs)

        if auc > best_auc:
            best_auc = auc
            best_params = params

    rf_model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        **best_params
    )

    return rf_model, best_params, best_auc


def tune_xgboost(X_train_scaled, y_train, X_val_scaled, y_val):
    np.random.seed(RANDOM_STATE)
    idx = np.random.choice(
        len(X_train_scaled),
        size=min(200000, len(X_train_scaled)),
        replace=False
    )
    X_tune = X_train_scaled[idx]
    y_tune = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

    scale_pos_weight = get_scale_pos_weight(y_train)

    best_auc = -1
    best_params = None

    for params in tqdm(list(ParameterGrid(XGB_GRID)), desc="Tuning XGBoost"):
        model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            **params
        )
        model.fit(X_tune, y_tune)
        probs = model.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, probs)

        if auc > best_auc:
            best_auc = auc
            best_params = params

    xgb_model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        **best_params
    )

    return xgb_model, best_params, best_auc, scale_pos_weight


def compare_advanced_models(X_train_scaled, y_train, X_val_scaled, y_val):
    baseline_model, baseline_result = train_baseline_logistic(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    rf_model, rf_params, rf_auc = tune_random_forest(
        X_train_scaled, y_train, X_val_scaled, y_val
    )
    xgb_model, xgb_params, xgb_auc, spw = tune_xgboost(
        X_train_scaled, y_train, X_val_scaled, y_val
    )

    rf_results = eval_binary(
        "Random Forest (tuned)",
        rf_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val
    )

    xgb_results = eval_binary(
        "XGBoost (tuned)",
        xgb_model,
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val
    )

    results_table = pd.DataFrame([
        baseline_result,
        rf_results,
        xgb_results
    ]).sort_values("Val_ROC_AUC", ascending=False).reset_index(drop=True)

    metadata = {
        "rf_params": rf_params,
        "rf_best_auc": rf_auc,
        "xgb_params": xgb_params,
        "xgb_best_auc": xgb_auc,
        "scale_pos_weight": spw
    }

    models = {
        "baseline_model": baseline_model,
        "rf_model": rf_model,
        "xgb_model": xgb_model
    }

    return results_table, models, metadata


def train_best_model_on_trainval(results_table, models, X_train_scaled, X_val_scaled, y_train, y_val):
    best_name = results_table.iloc[0]["Model"]

    if "Logistic Regression" in best_name:
        best_model = models["baseline_model"]
    elif "Random Forest" in best_name:
        best_model = models["rf_model"]
    else:
        best_model = models["xgb_model"]

    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = (
        pd.concat([y_train, y_val], axis=0)
        if hasattr(y_train, "iloc")
        else np.concatenate([y_train, y_val])
    )

    best_model.fit(X_trainval, y_trainval)
    return best_name, best_model


def train_all_models_on_trainval(models, X_train_scaled, X_val_scaled, y_train, y_val):
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = (
        pd.concat([y_train, y_val], axis=0)
        if hasattr(y_train, "iloc")
        else np.concatenate([y_train, y_val])
    )

    trained_models = {}
    for name, model in models.items():
        model.fit(X_trainval, y_trainval)
        trained_models[name] = model

    return trained_models