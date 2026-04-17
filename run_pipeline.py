import time
import pandas as pd

from src.utils import ensure_directories
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.splitting import chronological_split, scale_splits
from src.train import (
    compare_advanced_models,
    train_best_model_on_trainval,
    train_all_models_on_trainval
)
from src.evaluation import (
    evaluate_on_test,
    evaluate_multiple_models_on_test,
    create_final_model_summary,
    run_threshold_analysis,
    run_temporal_robustness_analysis
)
from src.unsupervised import run_pca_kmeans
from src.interpretation import (
    fit_interpretation_model,
    plot_feature_importance,
    plot_partial_dependence,
)
from src.config import TABLES_DIR


def log_step(step_name):
    print(f"\n{'=' * 60}")
    print(f">>> {step_name}")
    print(f"{'=' * 60}")
    return time.time()


def log_done(start_time, extra_message=None):
    elapsed = round(time.time() - start_time, 2)
    print(f"Done in {elapsed} seconds")
    if extra_message:
        print(extra_message)


def main():
    total_start = time.time()

    print("\nStarting end-to-end pipeline...")

    t = log_step("Creating output directories")
    ensure_directories()
    log_done(t)

    # 1. Download and load Kaggle data
    t = log_step("Downloading dataset from Kaggle and loading data")
    df, csv_path = load_data()
    log_done(
        t,
        extra_message=(
            f"Loaded data from: {csv_path}\n"
            f"Raw dataset shape: {df.shape}"
        ),
    )

    # 2. Preprocess and build target
    t = log_step("Preprocessing data and building target")
    df, threshold = preprocess_data(df)
    log_done(
        t,
        extra_message=(
            f"Post-preprocessing shape: {df.shape}\n"
            f"High-demand threshold (95th percentile of total_tx): {threshold:.4f}"
        ),
    )

    # 3. Feature engineering
    t = log_step("Engineering features")
    df_fe, base_features, engineered_features = engineer_features(df)
    log_done(
        t,
        extra_message=(
            f"Shape after feature engineering: {df_fe.shape}\n"
            f"Base feature count: {len(base_features)}\n"
            f"Engineered feature count: {len(engineered_features)}"
        ),
    )

    # 4. Base-feature split + scaling
    t = log_step("Creating chronological train/validation/test splits for base features")
    (
        train_df,
        val_df,
        test_df,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    ) = chronological_split(df_fe, base_features)

    scaler_base, X_train_scaled, X_val_scaled, X_test_scaled = scale_splits(
        X_train, X_val, X_test
    )

    log_done(
        t,
        extra_message=(
            f"Train shape: {X_train.shape}, Positive rate: {y_train.mean():.4f}\n"
            f"Validation shape: {X_val.shape}, Positive rate: {y_val.mean():.4f}\n"
            f"Test shape: {X_test.shape}, Positive rate: {y_test.mean():.4f}"
        ),
    )

    # 5. Train and compare supervised models
    t = log_step("Training baseline + advanced supervised models")
    validation_results, models, metadata = compare_advanced_models(
        X_train_scaled,
        y_train,
        X_val_scaled,
        y_val,
    )

    validation_results.to_csv(TABLES_DIR / "validation_model_comparison.csv", index=False)

    log_done(
        t,
        extra_message=(
            "Validation comparison table:\n"
            f"{validation_results.to_string(index=False)}\n\n"
            f"Saved: {TABLES_DIR / 'validation_model_comparison.csv'}"
        ),
    )

    # 6. Train all models on train+validation and evaluate on test
    t = log_step("Retraining all supervised models on train+validation and evaluating on test")
    trained_models = train_all_models_on_trainval(
        models,
        X_train_scaled,
        X_val_scaled,
        y_train,
        y_val,
    )

    test_model_map = {
        "Logistic Regression (baseline)": trained_models["baseline_model"],
        "Random Forest (tuned)": trained_models["rf_model"],
        "XGBoost (tuned)": trained_models["xgb_model"]
    }

    test_summary = evaluate_multiple_models_on_test(
        test_model_map,
        X_test_scaled,
        y_test
    )
    test_summary.to_csv(TABLES_DIR / "all_test_model_metrics.csv", index=False)

    final_summary = create_final_model_summary(validation_results, test_summary)

    log_done(
        t,
        extra_message=(
            "Test comparison table:\n"
            f"{test_summary.to_string(index=False)}\n\n"
            f"Saved: {TABLES_DIR / 'all_test_model_metrics.csv'}\n"
            f"Saved: {TABLES_DIR / 'final_model_summary.csv'}"
        ),
    )

    # 7. Evaluate best model in detail on test
    t = log_step("Running detailed test evaluation on best validation model")
    best_name, best_model = train_best_model_on_trainval(
        validation_results,
        models,
        X_train_scaled,
        X_val_scaled,
        y_train,
        y_val,
    )

    test_results, test_probs, test_preds = evaluate_on_test(
        best_name,
        best_model,
        X_test_scaled,
        y_test,
    )

    log_done(
        t,
        extra_message=(
            f"Best validation model: {best_name}\n"
            f"Detailed test results:\n{test_results.to_string(index=False)}\n\n"
            f"Saved: {TABLES_DIR / 'test_metrics.csv'}"
        ),
    )

    # 8. Threshold analysis
    t = log_step("Running threshold analysis on best model")
    threshold_df = run_threshold_analysis(
        best_name,
        best_model,
        X_test_scaled,
        y_test
    )
    log_done(
        t,
        extra_message=(
            f"Threshold analysis:\n{threshold_df.to_string(index=False)}\n\n"
            f"Saved: {TABLES_DIR / 'threshold_metrics.csv'}"
        ),
    )

    # 9. Temporal robustness
    t = log_step("Running temporal robustness analysis on best model")
    robustness_df = run_temporal_robustness_analysis(
        best_name,
        best_model,
        X_test_scaled,
        y_test
    )
    log_done(
        t,
        extra_message=(
            f"Temporal robustness:\n{robustness_df.to_string(index=False)}\n\n"
            f"Saved: {TABLES_DIR / 'temporal_robustness.csv'}"
        ),
    )

    # 10. Engineered-feature split for interpretation and unsupervised work
    t = log_step("Preparing engineered-feature datasets for interpretation and clustering")
    (
        train_fe_df,
        val_fe_df,
        test_fe_df,
        X_train_fe,
        y_train_fe,
        X_val_fe,
        y_val_fe,
        X_test_fe,
        y_test_fe,
    ) = chronological_split(df_fe, engineered_features)

    log_done(
        t,
        extra_message=(
            f"Engineered train shape: {X_train_fe.shape}\n"
            f"Engineered validation shape: {X_val_fe.shape}\n"
            f"Engineered test shape: {X_test_fe.shape}"
        ),
    )

    # 11. Interpretation model
    t = log_step("Fitting interpretation model and generating feature importance + PDPs")
    interpret_model = fit_interpretation_model(
        X_train_fe,
        y_train_fe,
        metadata["xgb_params"],
        metadata["scale_pos_weight"],
    )

    top_features = plot_feature_importance(
        interpret_model,
        engineered_features,
        top_n=15,
    )
    top_3_features = plot_partial_dependence(
        interpret_model,
        X_train_fe.astype(float),
        top_features,
    )

    log_done(
        t,
        extra_message=(
            "Top feature importances:\n"
            f"{top_features.to_string()}\n\n"
            f"Top PDP features: {top_3_features}"
        ),
    )

    # 12. Unsupervised learning
    t = log_step("Running PCA + KMeans clustering on engineered features")
    cluster_eval, cluster_summary, cluster_sizes, unsup_metadata = run_pca_kmeans(
        df_fe,
        engineered_features,
    )

    log_done(
        t,
        extra_message=(
            "Cluster evaluation:\n"
            f"{cluster_eval.to_string(index=False)}\n\n"
            "Cluster sizes:\n"
            f"{cluster_sizes.to_string()}\n\n"
            "Cluster summary:\n"
            f"{cluster_summary.to_string()}\n\n"
            f"Unsupervised metadata: {unsup_metadata}"
        ),
    )

    # 13. Save run metadata
    t = log_step("Saving run summary metadata")
    run_summary = {
        "source_csv_path": str(csv_path),
        "raw_rows_loaded": len(df),
        "rows_after_preprocessing_and_fe": len(df_fe),
        "high_demand_threshold": threshold,
        "best_validation_model": best_name,
        "rf_best_params": str(metadata["rf_params"]),
        "rf_best_auc": metadata["rf_best_auc"],
        "xgb_best_params": str(metadata["xgb_params"]),
        "xgb_best_auc": metadata["xgb_best_auc"],
        "scale_pos_weight": metadata["scale_pos_weight"],
        "best_k": unsup_metadata["best_k"],
        "pca_components_for_90pct_variance": unsup_metadata["n_components_90"],
        "pca_explained_variance": unsup_metadata["explained_variance"],
        "final_silhouette": unsup_metadata["final_silhouette"],
    }

    pd.DataFrame([run_summary]).to_csv(TABLES_DIR / "run_summary.csv", index=False)
    log_done(
        t,
        extra_message=f"Saved: {TABLES_DIR / 'run_summary.csv'}",
    )

    total_elapsed = round(time.time() - total_start, 2)
    print(f"\n{'=' * 60}")
    print("Pipeline completed successfully")
    print(f"Total runtime: {total_elapsed} seconds")
    print(f"Outputs saved under: {TABLES_DIR.parent}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()