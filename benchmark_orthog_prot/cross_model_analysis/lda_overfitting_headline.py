"""Headline graphic: in-sample LDA AUC (fit and scored on the same data) makes every
DL model look strong, but leave-one-out cross-validated AUC shows the combined-metric
LDA overfitting everywhere to some degree. Adding a third bar -- each model's own best
SINGLE metric's AUC (already an honest number, since a raw metric has no parameters
fit to the labels, so it needs no cross-validation) -- shows the combined-metric LDA
never earns its keep: the best single metric beats the LDA's cross-validated AUC in
all 8 model/dataset combinations, even the two (Boltz-2, ESMFold2 on DHD) where LDA
looks like it "generalizes."

Reads already-computed CSVs instead of recomputing: LDA numbers from lda_comparison.py's
per-dataset output, best-single-metric AUC from each project's own roc_auc.py leaderboard
CSV. This is a re-plot of existing numbers, not a new analysis. RIAM/FastRelax are
excluded per request; only the 4 DL models are shown.

New file -- does not modify or overwrite lda_comparison.py, roc_auc.py, or their outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
DATASETS = ["dhd", "cop"]
DATASET_TITLES = {"dhd": "DHD", "cop": "COP"}

# each model's own best-single-metric leaderboard CSV, per dataset (roc_auc.py's output)
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


# validated categorical triple (dataviz skill palette, slots 1, 7, 8)
IN_SAMPLE_COLOR = "#2a78d6"
BEST_METRIC_COLOR = "#4a3aa7"
CV_COLOR = "#e34948"
CHANCE_COLOR = "#898781"

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)

for ax, dataset in zip(axes, DATASETS):
    df = pd.read_csv(RESULTS_DIR / dataset / "lda_comparison.csv").set_index("model").loc[MODEL_ORDER]
    df["best_metric_name"], df["best_metric_auc"] = zip(*(best_single_metric_auc(m, dataset) for m in MODEL_ORDER))

    x = np.arange(len(MODEL_ORDER))
    width = 0.26
    ax.bar(x - width, df["in_sample_auc"], width, color=IN_SAMPLE_COLOR, label="LDA in-sample AUC")
    ax.bar(x, df["best_metric_auc"], width, color=BEST_METRIC_COLOR, label="Best single metric AUC")
    ax.bar(x + width, df["cv_auc"], width, color=CV_COLOR, label="LDA cross-validated AUC")
    ax.axhline(0.5, linestyle="--", linewidth=1, color=CHANCE_COLOR, zorder=0, label="Chance (AUC=0.5)")

    for xi, row in zip(x, df.itertuples()):
        ax.text(xi - width, row.in_sample_auc + 0.02, f"{row.in_sample_auc:.2f}",
                 ha="center", fontsize=8, color="#52514e")
        ax.text(xi, row.best_metric_auc + 0.02, f"{row.best_metric_auc:.2f}",
                 ha="center", fontsize=8.5, color="#0b0b0b", fontweight="bold")
        ax.text(xi + width, row.cv_auc + 0.02, f"{row.cv_auc:.2f}",
                 ha="center", fontsize=8.5, color="#0b0b0b", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER)
    ax.set_ylim(0, 1.12)
    ax.set_title(DATASET_TITLES[dataset], fontsize=12, color="#0b0b0b")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("AUC")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.98), frameon=False, fontsize=9.5)

fig.suptitle(
    "The best single metric beats the full multi-metric LDA combination —\n"
    "in every model, on both datasets",
    fontsize=14, fontweight="bold", y=1.12,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(RESULTS_DIR / "lda_overfitting_headline.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"saved {RESULTS_DIR / 'lda_overfitting_headline.png'}")
