"""PCA over the per-pair metrics: report the eigenvalues and how much variance each
component explains, then check whether the dominant variance directions separate
cognate from non-cognate pairs. Standardizes the metrics before decomposing, since
they're on very different scales. Runs once per project (cop and dhd)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pairing_utils import UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS, iter_datasets

MODEL_NAME = "ESMFold2"
ID_COL = "job_name"

for df, csv_stem, project, sep, results_dir in iter_datasets():
    pca_dir = results_dir / "pca"
    pca_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [c for c in df.columns if c not in UNWANTED_COLUMNS]

    # --- orient error-like metrics (lower-is-better -> higher-is-better) and standardize ---
    data = df[metric_columns].copy()
    for col in LOWER_IS_BETTER_COLUMNS:
        if col in data.columns:
            data[col] = -data[col]
    data.index = df[ID_COL]

    X = StandardScaler().fit_transform(data.values)  # rows = pairs, columns = metrics

    # --- eigen-decomposition of the covariance matrix across metrics ---
    cov_matrix = np.cov(X, rowvar=False)   # rowvar=False: columns are the variables (metrics)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # sort descending so PC1 is the highest-variance direction
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    pc_scores = X @ eigenvectors[:, :2]
    df['PC1'] = pc_scores[:, 0]
    df['PC2'] = pc_scores[:, 1]

    # --- report eigenvalues / variance explained ---
    variance_df = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(eigenvalues))],
        'eigenvalue': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'cumulative_variance_ratio': np.cumsum(explained_variance_ratio),
    })
    variance_df.to_csv(pca_dir / f"{csv_stem}_pca_variance.csv", index=False)
    print(f"--- {project} ---")
    print(variance_df.to_string(index=False))

    # --- 1. scree plot: variance explained per component ---
    plt.figure(figsize=(7, 5))
    components = range(1, len(eigenvalues) + 1)
    plt.bar(components, explained_variance_ratio, color='steelblue', label='Individual')
    plt.plot(components, np.cumsum(explained_variance_ratio), color='darkorange', marker='o', label='Cumulative')
    plt.xlabel("Principal Component")
    plt.ylabel("Fraction of Variance Explained")
    plt.title(f"Scree Plot — {MODEL_NAME} ({project})")
    plt.xticks(components)
    plt.legend()
    plt.tight_layout()
    plt.savefig(pca_dir / f"{csv_stem}_pca_scree.png", dpi=300)
    plt.close()

    # --- 2. PC1 vs PC2 scatter, colored by cognate/non-cognate ---
    plt.figure(figsize=(7, 6))
    for label, name, color in [(1, 'Cognate', 'steelblue'), (0, 'Non-cognate', 'firebrick')]:
        mask = df['cognate_interaction'] == label
        plt.scatter(df.loc[mask, 'PC1'], df.loc[mask, 'PC2'], label=name, alpha=0.6, color=color)
    plt.xlabel(f"PC1 ({explained_variance_ratio[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({explained_variance_ratio[1]:.1%} variance)")
    plt.title(f"PC1 vs PC2 by Cognate Status — {MODEL_NAME} ({project})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pca_dir / f"{csv_stem}_pca_scatter.png", dpi=300)
    plt.close()

    # --- 3. loadings bar chart: which metrics drive PC1 and PC2 ---
    loadings = pd.DataFrame(eigenvectors[:, :2], index=metric_columns, columns=['PC1', 'PC2'])
    loadings.to_csv(pca_dir / f"{csv_stem}_pca_loadings.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, pc in zip(axes, ['PC1', 'PC2']):
        ordered = loadings[pc].sort_values()
        ax.barh(ordered.index, ordered.values, color='steelblue')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_title(f"{pc} Loadings")
        ax.set_xlabel("Loading")
    plt.suptitle(f"Metric Loadings on Top 2 Components — {MODEL_NAME} ({project})")
    plt.tight_layout()
    plt.savefig(pca_dir / f"{csv_stem}_pca_loadings.png", dpi=300)
    plt.close()
