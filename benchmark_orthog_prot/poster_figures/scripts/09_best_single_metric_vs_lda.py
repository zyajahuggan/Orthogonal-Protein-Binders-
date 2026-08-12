"""Poster version of cross_model_analysis/lda_overfitting_headline.py: in-sample LDA
AUC vs. each model's own best single-metric AUC vs. leave-one-out cross-validated LDA
AUC, all 4 DL models, both datasets side by side in one figure. Reads already-computed
CSVs (lda_comparison.py's per-dataset output, each project's own roc_auc.py leaderboard)
instead of recomputing -- lda_comparison.py must be run first.

New file -- does not modify or overwrite lda_overfitting_headline.py or its outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "cross_model_analysis" / "results"

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "figure.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
DATASETS = ["dhd", "cop"]
DATASET_TITLES = {"dhd": "DHD", "cop": "COP"}

LEADERBOARD_PATHS = {
    "AF3": lambda ds: BASE / f"af3/results/{ds}/roc_auc/af3_auc_{ds}_leaderboard.csv",
    "Boltz-2": lambda ds: BASE / f"boltz/results/{ds}/roc_auc/boltz_auc_{ds}_leaderboard.csv",
    "Chai": lambda ds: BASE / f"chai/results/{ds}/roc_auc/chai_auc_{ds}_leaderboard.csv",
    "ESMFold2": lambda ds: BASE / f"esmfold2/results/{ds}/roc_auc/esmfold2_auc_{ds}_leaderboard.csv",
}


def best_single_metric_auc(model, dataset):
    leaderboard = pd.read_csv(LEADERBOARD_PATHS[model](dataset))
    top = leaderboard.sort_values("auc", ascending=False).iloc[0]
    return top["metric"], top["auc"]


IN_SAMPLE_COLOR = "#2a78d6"
BEST_METRIC_COLOR = "#4a3aa7"
CV_COLOR = "#e34948"
CHANCE_COLOR = "#898781"

fig, axes = plt.subplots(1, 2, figsize=(17, 9), sharey=True)

for ax, dataset in zip(axes, DATASETS):
    df = pd.read_csv(RESULTS_DIR / dataset / "lda_comparison.csv").set_index("model").loc[MODEL_ORDER]
    df["best_metric_name"], df["best_metric_auc"] = zip(*(best_single_metric_auc(m, dataset) for m in MODEL_ORDER))

    x = np.arange(len(MODEL_ORDER))
    width = 0.26
    ax.bar(x - width, df["in_sample_auc"], width, color=IN_SAMPLE_COLOR, label="LDA In-Sample AUC")
    ax.bar(x, df["best_metric_auc"], width, color=BEST_METRIC_COLOR, label="Best Single Metric AUC")
    ax.bar(x + width, df["cv_auc"], width, color=CV_COLOR, label="LDA Cross-Validated AUC")
    ax.axhline(0.5, linestyle="--", linewidth=1.5, color=CHANCE_COLOR, zorder=0, label="Chance (AUC=0.5)")

    for xi, row in zip(x, df.itertuples()):
        ax.text(xi - width, row.in_sample_auc + 0.02, f"{row.in_sample_auc:.2f}",
                ha="center", fontsize=13, color="#52514e")
        ax.text(xi, row.best_metric_auc + 0.02, f"{row.best_metric_auc:.2f}",
                ha="center", fontsize=14, color="#0b0b0b", fontweight="bold")
        ax.text(xi + width, row.cv_auc + 0.02, f"{row.cv_auc:.2f}",
                ha="center", fontsize=14, color="#0b0b0b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 1.12)
    ax.set_title(DATASET_TITLES[dataset])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("AUC")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=15)

fig.suptitle(
    "Best Single Metric Beats Full-Metric LDA Combination —\n"
    "Every Model, Both Datasets",
    fontsize=24, fontweight="bold", y=1.14,
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig(OUT_DIR / "best_single_metric_vs_lda.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"saved {OUT_DIR / 'best_single_metric_vs_lda.png'}")
