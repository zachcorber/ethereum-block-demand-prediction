import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.config import (
    PCA_VARIANCE_THRESHOLD,
    K_RANGE,
    RANDOM_STATE,
    SILHOUETTE_SAMPLE_SIZE,
    PLOT_SAMPLE_SIZE,
    FIGURES_DIR,
    TABLES_DIR
)

def run_pca_kmeans(df, feature_cols):
    df_unsup = df[feature_cols].copy().dropna().reset_index(drop=True)

    if df_unsup.empty:
        raise ValueError("No rows available for unsupervised learning after dropna().")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_unsup)

    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)

    explained_var = np.cumsum(pca_full.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained_var) + 1), explained_var, marker="o")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Cumulative Explained Variance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pca_cumulative_explained_variance.png")
    plt.close()

    n_components_90 = np.argmax(explained_var >= PCA_VARIANCE_THRESHOLD) + 1

    pca = PCA(n_components=n_components_90)
    X_pca = pca.fit_transform(X_scaled)

    k_values = list(K_RANGE)
    inertias = []
    sil_scores = []

    sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X_pca))
    sample_idx = np.random.choice(len(X_pca), sample_size, replace=False)

    for k in tqdm(k_values, desc="Evaluating K-Means"):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        sil = silhouette_score(X_pca[sample_idx], labels[sample_idx])
        sil_scores.append(sil)

    cluster_eval = pd.DataFrame({
        "k": k_values,
        "inertia": inertias,
        "silhouette_score": sil_scores
    })
    cluster_eval.to_csv(TABLES_DIR / "cluster_eval.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_elbow.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, sil_scores, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score (sampled)")
    plt.title("Silhouette Scores for K-Means")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_silhouette.png")
    plt.close()

    best_k = int(cluster_eval.loc[cluster_eval["silhouette_score"].idxmax(), "k"])

    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)

    cluster_plot_df = pd.DataFrame({
        "PC1": X_pca_2d[:, 0],
        "PC2": X_pca_2d[:, 1],
        "cluster": cluster_labels
    })

    plot_sample_size = min(PLOT_SAMPLE_SIZE, len(cluster_plot_df))
    plot_sample_df = cluster_plot_df.sample(plot_sample_size, random_state=RANDOM_STATE)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        plot_sample_df["PC1"],
        plot_sample_df["PC2"],
        c=plot_sample_df["cluster"],
        alpha=0.5
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"K-Means Clusters in 2D PCA Space (k={best_k})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "kmeans_clusters_2d.png")
    plt.close()

    final_sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(X_pca))
    final_sample_idx = np.random.choice(len(X_pca), final_sample_size, replace=False)
    final_sil = silhouette_score(X_pca[final_sample_idx], cluster_labels[final_sample_idx])

    df_cluster_profile = df_unsup.copy()
    df_cluster_profile["cluster"] = cluster_labels

    cluster_summary = df_cluster_profile.groupby("cluster").mean(numeric_only=True)
    cluster_sizes = df_cluster_profile["cluster"].value_counts().sort_index()

    cluster_summary.to_csv(TABLES_DIR / "cluster_summary.csv")
    cluster_sizes.to_csv(TABLES_DIR / "cluster_sizes.csv")

    metadata = {
        "best_k": best_k,
        "n_components_90": n_components_90,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "final_silhouette": float(final_sil)
    }

    return cluster_eval, cluster_summary, cluster_sizes, metadata