"""Static summary plots for the cross-model AUC comparison written by compare_models.py:
a grouped bar chart of each shared metric's AUC across models, and a per-metric heatmap of
pairwise AUC differences with bootstrap significance. Reads the existing model_comparison
CSVs -- no recomputation. Runs once per dataset (cop and dhd)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# same fixed order as compare_models.py's PROJECTS dict, so colors stay stable across runs
MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

for dataset in ["cop", "dhd"]:
    stats_dir = RESULTS_DIR / dataset / "auc_stats"
    df = pd.read_csv(stats_dir / "model_comparison.csv")
    metrics = list(df["metric"].unique())

    # --- grouped bar chart: AUC per model per metric ---
    model_auc = {m: {} for m in MODEL_ORDER}
    for _, row in df.iterrows():
        model_auc[row["model_a"]][row["metric"]] = row["auc_a"]
        model_auc[row["model_b"]][row["metric"]] = row["auc_b"]

    x = np.arange(len(metrics))
    width = 0.8 / len(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(MODEL_ORDER):
        aucs = [model_auc[model][m] for m in metrics]
        ax.bar(x + i * width, aucs, width, label=model, color=MODEL_COLORS[model])
    ax.set_xticks(x + width * (len(MODEL_ORDER) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax.set_ylabel("AUC")
    ax.set_title(f"AUC by Model and Metric — {dataset}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(stats_dir / f"model_auc_comparison_{dataset}.png", dpi=300)
    plt.close()

    # --- per-metric heatmap of pairwise AUC differences, annotated with significance ---
    for metric in metrics:
        sub = df[df["metric"] == metric]
        n = len(MODEL_ORDER)
        diff_matrix = np.full((n, n), np.nan)
        sig_matrix = np.zeros((n, n), dtype=bool)
        for _, row in sub.iterrows():
            i, j = MODEL_ORDER.index(row["model_a"]), MODEL_ORDER.index(row["model_b"])
            diff_matrix[i, j] = row["diff"]
            diff_matrix[j, i] = -row["diff"]
            sig_matrix[i, j] = sig_matrix[j, i] = row["significant"]

        vlim = max(np.nanmax(np.abs(diff_matrix)), 0.05)
        fig, ax = plt.subplots(figsize=(6, 5.5))
        im = ax.imshow(diff_matrix, cmap="RdBu", vmin=-vlim, vmax=vlim)
        ax.set_xticks(range(n))
        ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels(MODEL_ORDER)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                label = f"{diff_matrix[i, j]:+.2f}"
                if sig_matrix[i, j]:
                    label += "*"
                ax.text(j, i, label, ha="center", va="center", fontsize=9)
        ax.set_title(f"AUC(row) − AUC(col) — {metric} ({dataset})\n* = significant (95% CI excludes 0)",
                      fontsize=10)
        fig.colorbar(im, ax=ax, label="AUC diff", fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(stats_dir / f"model_diff_heatmap_{metric}_{dataset}.png", dpi=300)
        plt.close()
