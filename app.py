import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="COMP333 Project: Ethereum Block Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
ROOT = Path(".")
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Helpers
@st.cache_data
def load_csv(filename: str):
    path = TABLES_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None

def show_image(filename: str, caption: str = "", width="stretch"):
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, width=width)
    else:
        st.info(f"Missing figure: {filename}")

def show_df(df, title: str = None):
    if title:
        st.subheader(title)
    if df is not None:
        safe_df = df.copy()
        for col in safe_df.columns:
            if safe_df[col].dtype == "object":
                safe_df[col] = safe_df[col].astype(str)
        st.dataframe(safe_df, width="stretch")
    else:
        st.info("This table was not found in outputs/tables.")

def metric_card(label, value):
    st.metric(label, value)

# Load outputs
validation_model_comparison = load_csv("validation_model_comparison.csv")
all_test_model_metrics = load_csv("all_test_model_metrics.csv")
final_model_summary = load_csv("final_model_summary.csv")
test_metrics = load_csv("test_metrics.csv")
threshold_metrics = load_csv("threshold_metrics.csv")
temporal_robustness = load_csv("temporal_robustness.csv")
cluster_eval = load_csv("cluster_eval.csv")
cluster_sizes = load_csv("cluster_sizes.csv")
cluster_summary = load_csv("cluster_summary.csv")
run_summary = load_csv("run_summary.csv")
top_feature_importance = load_csv("top_feature_importance.csv")

st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choose a section",
    [
        "Initial Data",
        "Supervised Learning",
        "Feature Engineering",
        "Unsupervised Learning",
        "Interpretation",
        "Pipeline & Conclusion"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Author")
st.sidebar.markdown(
    """
1. Zachary Corber
"""
)

st.title("COMP333 Project: Ethereum Block Data")

# Section: Initial Data
if section == "Initial Data":
    st.header("Initial Data")

    st.markdown(
        """
This section summarizes the beginning of the project: data retrieval, dataset characteristics,
wrangling and cleaning, exploratory analysis, and the research questions that guided the rest
of the work.
"""
    )

    if run_summary is not None and len(run_summary) > 0:
        row = run_summary.iloc[0]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Raw Rows Loaded", f"{int(row['raw_rows_loaded']):,}")
        with c2:
            metric_card("Rows After Cleaning + FE", f"{int(row['rows_after_preprocessing_and_fe']):,}")
        with c3:
            metric_card("Raw Cols Loaded", "16")
        with c4:
            metric_card("Dataset Size (Uncompressed)", "1.156 GB")

        with st.expander("Show full run summary table"):
            show_df(run_summary)
            
    st.markdown("## Key Columns in the Dataset")
    
    columns_data = [
        ("total_tx", "Total number of transactions in the block"),
        ("block_size", "Size of the block"),
        ("gas_limit", "Maximum gas allowed in the block"),
        ("gas_used (cost)", "Total gas used"),
        ("gas_avg_price", "Average gas price"),
        ("block_time_in_sec", "Time taken to mine the block in seconds"),
        ("hour", "Hour of the day when the block was mined"),
        ("high_demand", "Indicator for high demand block"),
        ("target", "Binary target indicating if the next block is high demand"),
    ]
    
    df_cols = pd.DataFrame(columns_data, columns=["Column Name", "Description"])
    st.dataframe(df_cols, width="stretch")

    st.markdown("## Data Retrieval")
    st.markdown(
        """
The final pipeline retrieves the Ethereum dataset directly from Kaggle rather than relying on a manually downloaded file.
This improves reproducibility and ensures the project can be rerun on a new machine with the same setup.
"""
    )

    if run_summary is not None and "source_csv_path" in run_summary.columns:
        st.code(str(run_summary.iloc[0]["source_csv_path"]))

    st.markdown("## Dataset and Cleaning")
    st.markdown(
        """
The notebook workflow performs the following early-stage cleaning and preparation steps:

- drops irrelevant columns,
- sorts the data by block height,
- converts timestamps into datetime format,
- removes duplicates,
- defines a high-demand flag,
- and constructs the predictive target using a one-step shift.

This creates a supervised target that asks whether the **next** Ethereum block will experience unusually high demand.
"""
    )

    st.markdown("## Exploratory Data Analysis (EDA)")
    st.markdown(
        """
The EDA stage was used to understand the data distribution, identify skew and correlation structure,
and motivate the modeling choices used later in the project.
"""
    )
    
    st.markdown("## Research Questions")

    col1, col2 = st.columns(2)
    with col1:
        st.info("""
**Supervised Learning Question**

Can we predict whether the next Ethereum block will have unusually high demand?
""")
    with col2:
        st.info("""
**Unsupervised Learning Question**

Can we identify distinct Ethereum activity regimes using clustering?
""")

# Section: Supervised Learning
elif section == "Supervised Learning":
    st.header("Supervised Learning")

    st.markdown(
        """
This section presents the supervised learning workflow and the main predictive results.

The project compares:
- **Logistic Regression** as a baseline,
- **Random Forest** as a tree-based ensemble,
- **XGBoost** as the strongest tuned nonlinear model.

The target is whether the **next Ethereum block** falls into a high-demand regime.
"""
    )

    st.markdown("## Overall Model Comparison")
    show_df(validation_model_comparison, "Validation Model Comparison")
    show_df(all_test_model_metrics, "Test Model Comparison")
    show_df(final_model_summary, "Final Combined Model Summary")

    st.markdown("---")
    st.markdown("## Validation Curves by Model")

    st.markdown("### 1. Logistic Regression (Baseline)")
    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "logistic_regression_baseline_val_roc.png",
            "Logistic Regression - Validation ROC Curve",
            width=500
        )
    with col2:
        show_image(
            "logistic_regression_baseline_val_pr.png",
            "Logistic Regression - Validation Precision-Recall Curve",
            width=500
        )

    show_image(
        "logistic_regression_baseline_val_confusion_matrix.png",
        "Logistic Regression - Validation Confusion Matrix",
        width=500,
    )

    st.markdown(
        """
**Interpretation:**  
Logistic Regression provides the baseline reference point. Its performance shows that the problem contains
meaningful predictive structure even before using more complex nonlinear models.
"""
    )

    st.markdown("### 2. Random Forest")
    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "random_forest_tuned_val_roc.png",
            "Random Forest - Validation ROC Curve",
            width=500
        )
    with col2:
        show_image(
            "random_forest_tuned_val_pr.png",
            "Random Forest - Validation Precision-Recall Curve",
            width=500
        )

    show_image(
        "random_forest_tuned_val_confusion_matrix.png",
        "Random Forest - Validation Confusion Matrix",
        width=500,
    )

    st.markdown(
        """
**Interpretation:**  
Random Forest improves over the baseline by capturing nonlinear interactions and threshold effects,
but it still does not outperform XGBoost on the validation set.
"""
    )

    st.markdown("### 3. XGBoost")
    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "xgboost_tuned_val_roc.png",
            "XGBoost - Validation ROC Curve",
            width=500
        )
    with col2:
        show_image(
            "xgboost_tuned_val_pr.png",
            "XGBoost - Validation Precision-Recall Curve",
            width=500
        )

    show_image(
        "xgboost_tuned_val_confusion_matrix.png",
        "XGBoost - Validation Confusion Matrix",
        width=500,
    )

    st.markdown(
        """
**Interpretation:**  
XGBoost achieved the strongest validation performance overall, which is why it was selected as the
best validation model in the final workflow.
"""
    )

    st.markdown("---")
    st.markdown("## Best Model: Detailed Test Evaluation")

    show_df(test_metrics, "Best Model Test Metrics")

    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "xgboost_tuned_test_roc.png",
            "XGBoost - Test ROC Curve",
            width=500
        )
        show_image(
            "xgboost_tuned_test_confusion_matrix.png",
            "XGBoost - Test Confusion Matrix",
            width=500
        )
    with col2:
        show_image(
            "xgboost_tuned_test_pr.png",
            "XGBoost - Test Precision-Recall Curve",
            width=500
        )
        show_image(
            "xgboost_tuned_test_calibration.png",
            "XGBoost - Test Calibration Curve",
            width=500
        )

    st.markdown("---")
    st.markdown("## Interpretation and Analysis")

    left_col, right_col = st.columns([1.1, 1])

    with left_col:
        st.markdown(
            """
### What the results show

The validation results identify **XGBoost** as the strongest overall model. This indicates that nonlinear
relationships and interactions between Ethereum block features matter for prediction.

At the same time, the test comparison shows that **Logistic Regression remains competitive**, which suggests
that the dataset also contains important linear structure. This is a useful finding because it shows that
the problem is not purely nonlinear.

The results also highlight a classic imbalanced-classification tradeoff:
- **recall is high**, meaning the model catches many high-demand blocks,
- but **precision is low**, meaning many predicted spikes are false positives.

This means the models are useful for identifying risky periods, but the operating threshold must be chosen
carefully depending on whether the priority is catching more spikes or reducing false alarms.
"""
        )

    with right_col:
        st.markdown("### Key Tables")
        if validation_model_comparison is not None:
            st.markdown("**Validation Ranking**")
            st.dataframe(validation_model_comparison, width="stretch")

        if all_test_model_metrics is not None:
            st.markdown("**Test Ranking**")
            st.dataframe(all_test_model_metrics, width="stretch")

# Section: Feature Engineering
elif section == "Feature Engineering":
    st.header("Feature Engineering")

    st.markdown(
        """
Feature engineering was one of the most important improvements in the project.

Beyond the original block variables, the pipeline adds:
- ratio-based features,
- log transforms,
- lag features,
- rolling-window transaction features.

These engineered features help capture the short-term temporal dynamics of Ethereum block activity.
"""
    )

    show_df(top_feature_importance, "Top Feature Importances")

    st.markdown("## Feature Importance")
    show_image(
        "feature_importance.png",
        "Feature Importance for the Interpretation Model",
        width=500
    )

    st.markdown("## Partial Dependence Plots")

    pdp_options = {
        "20-block rolling transactions": "pdp_total_tx_roll20.png",
        "5-block rolling transactions": "pdp_total_tx_roll5.png",
        "Log total transactions": "pdp_log_total_tx.png"
    }

    selected_pdp = st.selectbox(
        "Choose a partial dependence plot to inspect",
        list(pdp_options.keys())
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        show_image(
            pdp_options[selected_pdp],
            f"Partial Dependence: {selected_pdp}",
            width=500
        )

    with col2:
        st.markdown(
            """
### Explanation

The engineered features reveal that recent transaction behavior is highly predictive.

In particular, the most important features are dominated by:
- rolling transaction counts,
- lagged transaction information,
- and transformed transaction-volume features.

This shows that high-demand Ethereum blocks are not isolated random events. Instead, they tend to occur during short-lived periods of elevated network activity.

That is why the feature engineering stage adds real value beyond using only raw block-level variables.
"""
        )

    st.markdown("## Main Takeaway")
    st.success(
    """
Feature engineering significantly improves the model by capturing short-term temporal structure.
The strongest predictors come from recent transaction history, especially rolling and lagged transaction features.

This directly improves our Ethereum predictions by allowing the model to detect emerging spikes in network activity, 
making it more effective at identifying when a high-demand block is likely to occur next.
"""
)

# Section: Unsupervised Learning
elif section == "Unsupervised Learning":
    st.header("Unsupervised Learning")

    st.markdown(
        """
This section applies **unsupervised learning** to identify natural patterns in Ethereum block activity.

The goal is to determine whether blocks cluster into distinct activity regimes without using the target label.
"""
    )

    st.markdown("## Step 1: Dimensionality Reduction (PCA)")
    show_image(
        "pca_cumulative_explained_variance.png",
        "PCA Cumulative Explained Variance",
        width=500
    )

    st.markdown(
        """
PCA is used to reduce the dimensionality of the engineered feature space while preserving most of the variance.

This allows clustering to focus on the most important underlying structure in the data.
"""
    )

    st.markdown("## Step 2: Choosing Number of Clusters (K)")

    col1, col2 = st.columns(2)
    with col1:
        show_image(
            "kmeans_elbow.png",
            "Elbow Method",
            width=500
        )
    with col2:
        show_image(
            "kmeans_silhouette.png",
            "Silhouette Score",
            width=500
        )

    st.markdown(
        """
Two methods were used to determine the optimal number of clusters:

- **Elbow Method:** looks for diminishing returns in inertia  
- **Silhouette Score:** measures how well-separated clusters are  

### Final Choice: **k = 2**
"""
    )

    st.success(
        """
The best clustering solution is obtained at **k = 2**, which is the value used in all final results.
"""
    )

    st.markdown("## Step 3: Cluster Visualization")

    show_image(
        "kmeans_clusters_2d.png",
        "Clusters in 2D PCA Space",
        width=500
    )

    st.markdown("## Cluster Results")

    col1, col2 = st.columns(2)
    with col1:
        show_df(cluster_sizes, "Cluster Sizes")
    with col2:
        show_df(cluster_eval, "Cluster Evaluation Metrics")

    show_df(cluster_summary, "Cluster Summary")

    st.markdown("## Interpretation")

    st.markdown(
        """
The clustering results reveal **two distinct Ethereum activity regimes**:

### Cluster 0: Lower Activity
- lower transaction counts  
- smaller block sizes  
- lower gas usage  

### Cluster 1: Higher Activity
- higher transaction counts  
- larger blocks  
- increased gas usage  
- stronger rolling transaction signals  

These clusters represent different operating states of the Ethereum network.
"""
    )

    st.markdown("## Main Takeaway")

    st.success(
        """
Ethereum block activity is not random — it naturally separates into two distinct regimes.

This provides valuable insight for prediction because it confirms that high-demand blocks occur within
broader high-activity periods, making them more predictable when recent network activity is taken into account.
"""
    )

# Section: Interpretation
elif section == "Interpretation":
    st.header("Interpretation")

    st.markdown(
        """
This section explains **why the final model was chosen**, how its performance should be interpreted,
how sensitive it is to threshold choice, how stable it is over time, and what ethical and practical
limitations should be kept in mind.
"""
    )

    st.markdown("## Why XGBoost Was Selected as the Best Model")

    st.success(
        """
XGBoost was selected as the best overall model because it achieved the strongest **validation performance**
while also capturing nonlinear relationships that simpler models could not represent as effectively.
"""
    )

    st.markdown(
        """
More specifically, XGBoost was chosen because it:

- achieved the **highest validation ROC-AUC** among the compared models,
- performed strongly on **PR-AUC**, which is important for this imbalanced classification problem,
- handled complex nonlinear relationships between Ethereum block features,
- and benefited the most from the engineered temporal and rolling features.

Even though Logistic Regression remained competitive on the held-out test set, XGBoost was still selected
as the best model because it provided the strongest overall validation evidence and the richest interpretation
of nonlinear structure in the data.
"""
    )

    st.markdown("## Feature-Based Interpretation")

    col1, col2 = st.columns([1, 1])
    with col1:
        show_image(
            "feature_importance.png",
            "XGBoost Feature Importance",
            width=500
        )
    with col2:
        st.markdown(
            """
### What the feature importance shows

The most important predictors are dominated by:
- rolling transaction activity,
- recent transaction history,
- transformed transaction-volume features.

This means that short-term Ethereum network behavior is highly informative for predicting whether the next block
will fall into a high-demand regime.

In practical terms, this shows that demand spikes are not isolated random events. They tend to emerge from
broader periods of elevated network activity, which the model is able to detect.
"""
        )

    st.markdown("## Threshold Analysis")

    show_df(threshold_metrics, "Threshold Metrics")

    threshold_choice = None
    if threshold_metrics is not None and "threshold" in threshold_metrics.columns:
        threshold_choice = st.select_slider(
            "Select a threshold",
            options=threshold_metrics["threshold"].tolist(),
            value=0.5
        )
        selected_row = threshold_metrics[threshold_metrics["threshold"] == threshold_choice]

        st.markdown("### Selected Threshold Performance")
        st.dataframe(selected_row, width="stretch")

    col1, col2 = st.columns([1, 1])
    with col1:
        show_image(
            "threshold_tradeoff.png",
            "Threshold Tradeoff Plot",
            width=500
        )
    with col2:
        st.markdown(
            """
### What the threshold analysis shows

The threshold analysis reveals a strong **precision-recall tradeoff**:

- at lower thresholds, recall is extremely high,
- but precision remains low,
- while at higher thresholds, precision improves and recall declines.

This means the model can be tuned depending on the goal:
- if the goal is to **catch as many spikes as possible**, use a lower threshold,
- if the goal is to **reduce false alarms**, use a higher threshold.

So the default threshold of 0.50 is not the only valid operating point.
"""
        )

    st.markdown("## Temporal Robustness")

    show_df(temporal_robustness, "Temporal Robustness Metrics")

    col1, col2 = st.columns([1, 1])
    with col1:
        show_image(
            "temporal_robustness.png",
            "Temporal Robustness Across Chronological Test Chunks",
            width=500
        )
    with col2:
        st.markdown(
            """
### What the robustness analysis shows

Performance varies across different chronological segments of the test set.
This indicates that the predictive difficulty of the task changes over time.

That matters because Ethereum activity is not static:
- transaction behavior changes,
- demand regimes shift,
- and model quality is not perfectly constant across periods.

This supports the need for careful monitoring and possible retraining if the system were used in practice.
"""
        )

    st.markdown("## Comprehensive Evaluation Summary")

    st.markdown(
        """
The final evaluation goes beyond a single metric and includes:

- held-out test performance,
- baseline versus advanced model comparison,
- threshold sensitivity,
- temporal robustness analysis,
- calibration-aware evaluation using Brier score,
- and model interpretation through importance and dependence analysis.

This is important because a single average test score does not fully describe model behavior,
especially in an imbalanced and time-varying prediction problem like Ethereum block demand.
"""
    )

    st.markdown("## Ethics, Limitations, and Future Work")

    st.markdown(
        """
### Ethics and Responsible Use
Although this project does not model human personal data directly, responsible interpretation still matters.
Predictions about Ethereum demand should not be treated as certain or causal, and they should not be used
without understanding their uncertainty and limitations.

### Bias and Fairness
Bias in this context appears as **time-based bias** rather than demographic bias.
A model trained on one period of Ethereum activity may perform differently in another period, which creates
uneven reliability across time.

### Privacy
The analysis uses public blockchain data at the block level.
It does not attempt deanonymization, identity inference, or address-level behavioral profiling.

### Limitations
- the high-demand label is defined using a percentile threshold and is therefore a proxy,
- strong class imbalance affects precision and recall,
- omitted variables such as mempool conditions or external market events may matter,
- and the model may drift as Ethereum usage patterns evolve over time.

### Future Work
Possible future improvements include:
- richer sequence or time-series models,
- dynamic threshold tuning,
- probability calibration improvements,
- drift detection and retraining,
- and inclusion of additional external signals.
"""
    )

    st.markdown("## Main Takeaway")

    st.success(
        """
XGBoost was selected because it provided the strongest overall validation evidence and best captured the nonlinear,
short-term activity patterns that drive Ethereum demand spikes.

At the same time, the interpretation results show that model performance depends on threshold choice and changes over time,
so predictions should be used as informative risk signals rather than perfect forecasts.
"""
    )

# Section: Pipeline & Conclusion
elif section == "Pipeline & Conclusion":
    st.header("Pipeline & Conclusion")

    st.markdown(
        """
This section summarizes the final pipeline and provides direct answers to the research questions,
based on all previous analysis.
"""
    )

    # -----------------------
    # Pipeline Summary
    # -----------------------
    st.markdown("## Final Pipeline Overview")

    st.markdown(
        """
The final system is a fully reproducible end-to-end pipeline that:

1. retrieves Ethereum block data from Kaggle  
2. performs wrangling and cleans the dataset  
3. engineers lag and rolling features  
4. performs train/validation/test splitting  
5. trains and compares multiple models  
6. evaluates performance with comprehensive metrics  
7. performs interpretation and clustering analysis  
8. saves all outputs automatically  

This structure ensures consistency, reproducibility, and alignment with real-world data workflows.
"""
    )

    st.markdown("## Research Questions: Final Answers")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Supervised Learning Question")

        st.success(
            """
Yes — it is possible to predict high-demand Ethereum blocks with meaningful accuracy.
"""
        )

        st.markdown(
            """
The models achieve performance above random, with XGBoost providing the strongest validation results.

However:
- precision remains low due to class imbalance  
- recall is high, meaning most spikes are detected  
- performance varies across time  

This means the model is effective at identifying **risk of demand spikes**, but predictions should
be interpreted as a probability rather than as exact results.
"""
        )

    with col2:
        st.markdown("### Unsupervised Learning Question")

        st.success(
            """
Yes — Ethereum activity naturally separates into distinct regimes.
"""
        )

        st.markdown(
            """
Clustering consistently identifies **two groups (k = 2)**:

- a lower-activity regime  
- a higher-activity regime  

This confirms that Ethereum network behavior is not uniform and instead operates in broader activity states.

This insight supports the supervised results, because high-demand blocks tend to occur within
these higher-activity regimes.
"""
        )

    st.markdown("## Key Insights")

    st.markdown(
        """
- Short-term transaction activity is the strongest predictor of future demand  
- Feature engineering significantly improves model performance  
- XGBoost captures nonlinear patterns more effectively than simpler models  
- Model performance depends strongly on threshold choice  
- Prediction quality varies over time due to changing network behavior  
- Ethereum activity is structured into distinct regimes rather than random  
"""
    )

    st.markdown("## Final Conclusion")

    st.success(
        """
This project demonstrates that Ethereum block demand can be predicted using machine learning,
especially when incorporating engineered temporal features.

While predictions are not perfect, the model provides useful signals for identifying periods of
elevated network activity.

Combined with clustering insights, the results show that Ethereum operates in structured activity regimes,
making both prediction and pattern discovery meaningful.
"""
    )
    st.markdown("## When Is Ethereum Congestion Highest?")

    st.success(
    """
Ethereum congestion is highest during periods of sustained elevated transaction activity,
rather than isolated spikes.
"""
)

    st.markdown(
    """
The analysis shows that high-demand blocks tend to occur when:

- recent transaction counts are already elevated  
- rolling transaction features (e.g., 5-block and 20-block averages) are high  
- gas usage and block size are consistently increased  

This means congestion builds over short time windows and is not purely random.

These findings are supported by:
- feature importance (rolling transaction features dominate),
- partial dependence plots (increasing trends with transaction activity),
- and clustering results (high-activity regime identified as a separate cluster).

Overall, Ethereum congestion is most likely during **short-term periods of sustained network activity**,
rather than single isolated events.
"""
)