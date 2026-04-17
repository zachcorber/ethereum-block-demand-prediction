import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, XGB_N_ESTIMATORS, FIGURES_DIR, TABLES_DIR

def fit_interpretation_model(X_train_fe, y_train_fe, params_for_xgb, scale_pos_weight):
    model = XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        **params_for_xgb
    )
    model.fit(X_train_fe, y_train_fe)
    return model

def plot_feature_importance(model, feature_names, top_n=15):
    feature_importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    top_features = feature_importance.head(min(top_n, len(feature_importance)))

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances - XGBoost")
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png")
    plt.close()

    top_features.to_csv(TABLES_DIR / "top_feature_importance.csv", header=["importance"])
    return top_features

def plot_partial_dependence(model, X_train_fe, top_features):
    top_3_features = top_features.index[:3].tolist()

    for feat in top_3_features:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        PartialDependenceDisplay.from_estimator(
            model,
            X_train_fe,
            [feat],
            ax=ax
        )
        plt.title(f"Partial Dependence Plot: {feat}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"pdp_{feat}.png")
        plt.close()

    return top_3_features