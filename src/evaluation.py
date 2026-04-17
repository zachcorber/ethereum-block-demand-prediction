import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

from src.config import FIGURES_DIR, TABLES_DIR, DEFAULT_THRESHOLD, THRESHOLD_GRID, ROBUSTNESS_CHUNKS


def _safe_name(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def eval_binary(name, model, X_tr, y_tr, X_va, y_va, threshold=DEFAULT_THRESHOLD, plot=True):
    model.fit(X_tr, y_tr)

    va_probs = model.predict_proba(X_va)[:, 1]
    va_pred = (va_probs >= threshold).astype(int)

    auc = roc_auc_score(y_va, va_probs)
    pr_auc = average_precision_score(y_va, va_probs)
    prec = precision_score(y_va, va_pred, zero_division=0)
    rec = recall_score(y_va, va_pred, zero_division=0)
    f1 = f1_score(y_va, va_pred, zero_division=0)
    cm = confusion_matrix(y_va, va_pred)

    print(f"\n===== {name} (Validation) =====")
    print("ROC-AUC:", round(auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_va, va_pred, zero_division=0))

    safe_name = _safe_name(name)

    if plot:
        fpr, tpr, _ = roc_curve(y_va, va_probs)
        plt.figure(figsize=(8, 5))
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name} (Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{safe_name}_val_roc.png")
        plt.close()

        p, r, _ = precision_recall_curve(y_va, va_probs)
        plt.figure(figsize=(8, 5))
        plt.plot(r, p, label=f"{name} (PR-AUC={pr_auc:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {name} (Validation)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"{safe_name}_val_pr.png")
        plt.close()

        plot_confusion_matrix(
            cm,
            f"Confusion Matrix - {name} (Validation)",
            FIGURES_DIR / f"{safe_name}_val_confusion_matrix.png"
        )

    return {
        "Model": name,
        "Threshold": threshold,
        "Val_ROC_AUC": auc,
        "Val_PR_AUC": pr_auc,
        "Val_Precision": prec,
        "Val_Recall": rec,
        "Val_F1": f1
    }


def evaluate_on_test(name, model, X_test, y_test, threshold=DEFAULT_THRESHOLD):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    brier = brier_score_loss(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print(f"\n===== {name} (Test) =====")
    print("ROC-AUC:", round(auc, 4))
    print("PR-AUC:", round(pr_auc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1:", round(f1, 4))
    print("Brier Score:", round(brier, 4))
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, preds, zero_division=0))

    safe_name = _safe_name(name)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name} (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{safe_name}_test_roc.png")
    plt.close()

    # PR
    p, r, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(8, 5))
    plt.plot(r, p, label=f"{name} (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name} (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{safe_name}_test_pr.png")
    plt.close()

    # Confusion matrix
    plot_confusion_matrix(
        cm,
        f"Confusion Matrix - {name} (Test)",
        FIGURES_DIR / f"{safe_name}_test_confusion_matrix.png"
    )

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    plt.figure(figsize=(8, 5))
    plt.plot(prob_pred, prob_true, marker="o", label=name)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {name} (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{safe_name}_test_calibration.png")
    plt.close()

    results = pd.DataFrame([{
        "Model": name,
        "Threshold": threshold,
        "Test_ROC_AUC": auc,
        "Test_PR_AUC": pr_auc,
        "Test_Precision": prec,
        "Test_Recall": rec,
        "Test_F1": f1,
        "Test_Brier": brier
    }])
    results.to_csv(TABLES_DIR / "test_metrics.csv", index=False)

    return results, probs, preds


def evaluate_multiple_models_on_test(model_map, X_test, y_test, threshold=DEFAULT_THRESHOLD):
    rows = []

    for model_name, model in model_map.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)

        row = {
            "Model": model_name,
            "Threshold": threshold,
            "Test_ROC_AUC": roc_auc_score(y_test, probs),
            "Test_PR_AUC": average_precision_score(y_test, probs),
            "Test_Precision": precision_score(y_test, preds, zero_division=0),
            "Test_Recall": recall_score(y_test, preds, zero_division=0),
            "Test_F1": f1_score(y_test, preds, zero_division=0),
            "Test_Brier": brier_score_loss(y_test, probs)
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Test_ROC_AUC", ascending=False).reset_index(drop=True)
    return df


def create_final_model_summary(validation_df, test_df):
    merged = validation_df.merge(test_df, on=["Model", "Threshold"], how="left")
    merged.to_csv(TABLES_DIR / "final_model_summary.csv", index=False)
    return merged


def run_threshold_analysis(name, model, X_test, y_test, thresholds=None):
    if thresholds is None:
        thresholds = THRESHOLD_GRID

    probs = model.predict_proba(X_test)[:, 1]
    rows = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0)
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df.to_csv(TABLES_DIR / "threshold_metrics.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
    plt.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Threshold Tradeoff - {name} (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_tradeoff.png")
    plt.close()

    return threshold_df


def run_temporal_robustness_analysis(name, model, X_test, y_test, n_chunks=ROBUSTNESS_CHUNKS):
    probs = model.predict_proba(X_test)[:, 1]

    y_test = np.array(y_test)
    probs = np.array(probs)

    chunk_size = len(y_test) // n_chunks
    rows = []

    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else len(y_test)

        y_chunk = y_test[start:end]
        p_chunk = probs[start:end]

        if len(np.unique(y_chunk)) < 2:
            roc_auc = np.nan
            pr_auc = np.nan
        else:
            roc_auc = roc_auc_score(y_chunk, p_chunk)
            pr_auc = average_precision_score(y_chunk, p_chunk)

        rows.append({
            "chunk": i + 1,
            "rows": len(y_chunk),
            "positive_rate": float(np.mean(y_chunk)),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc
        })

    robustness_df = pd.DataFrame(rows)
    robustness_df.to_csv(TABLES_DIR / "temporal_robustness.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(robustness_df["chunk"], robustness_df["roc_auc"], marker="o", label="ROC-AUC")
    plt.plot(robustness_df["chunk"], robustness_df["pr_auc"], marker="o", label="PR-AUC")
    plt.xlabel("Chronological Test Chunk")
    plt.ylabel("Metric Value")
    plt.title(f"Temporal Robustness - {name} (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temporal_robustness.png")
    plt.close()

    return robustness_df