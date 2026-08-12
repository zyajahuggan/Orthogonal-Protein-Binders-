"""AUC comparison extended with Rosetta's two scoring variants (RIAM, FastRelax) pooled
across all 4 spm structure sets, alongside best_metric_comparison.py's AF3/Boltz-2/Chai/
ESMFold2 bars. New file -- does not modify or overwrite best_metric_comparison.py,
best_metric_all_datasets.py, or their outputs.

Rosetta re-scores each DL model's predicted structures rather than generating its own, so
there's no single "Rosetta" metrics file -- there's one riam_{spm}_{dataset}_combined.csv
and one fast_relax_riam_{spm}_{dataset}_combined.csv per spm (af3/boltz/chai/esmfold2).
To get one RIAM bar and one FastRelax bar, the 4 spm files per variant are concatenated
into one pooled dataframe and AUC is computed once per candidate metric on the pooled
(score, cognate_status) pairs -- valid for AUC since it only needs labeled scores, not
matched cognate/competitor pairs (unlike win-rate/margin, where pooling raw rows would
wrongly match a cognate row from one spm's structures against a competitor row from
another spm's -- see best_winrate_comparison_with_rosetta.py and
best_margin_comparison_with_rosetta.py for how those two handle it instead).

ROSETTA_LOWER_IS_BETTER/ROSETTA_UNWANTED are copied from py_rosetta/scripts/analysis/
pairing_utils.py, same cross-project duplication precedent as py_rosetta/scripts/analysis/
riam_vs_own_metric_comparison.py.

Produces both a per-dataset 6-bar chart (best_metric_comparison.py's format) and a
combined-both-datasets grouped-bar chart (best_metric_all_datasets.py's format), since
computing the pooled Rosetta AUC needs only one pass over the data either way.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROSETTA_METRICS_DIR = BASE / "py_rosetta" / "metrics" / "metrics"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2", "RIAM", "FastRelax"]
MODEL_COLORS = {
    "AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300",
    "RIAM": "#8456ce", "FastRelax": "#c0392b",
}
DATASET_ORDER = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}
SPMS = ["af3", "boltz", "chai", "esmfold2"]
ROSETTA_VARIANTS = {"RIAM": "riam", "FastRelax": "fast_relax_riam"}

# leaderboard CSV path per model per dataset -- each is already sorted best-to-worst by AUC
LEADERBOARDS = {
    "AF3": {
        "cop": BASE / "af3/results/cop/roc_auc/af3_auc_cop_leaderboard.csv",
        "dhd": BASE / "af3/results/dhd/roc_auc/af3_auc_dhd_leaderboard.csv",
    },
    "Boltz-2": {
        "cop": BASE / "boltz/results/cop/roc_auc/boltz_auc_cop_leaderboard.csv",
        "dhd": BASE / "boltz/results/dhd/roc_auc/boltz_auc_dhd_leaderboard.csv",
    },
    "Chai": {
        "cop": BASE / "chai/results/cop/roc_auc/chai_auc_cop_leaderboard.csv",
        "dhd": BASE / "chai/results/dhd/roc_auc/chai_auc_dhd_leaderboard.csv",
    },
    "ESMFold2": {
        "cop": BASE / "esmfold2/results/cop/roc_auc/esmfold2_auc_cop_leaderboard.csv",
        "dhd": BASE / "esmfold2/results/dhd/roc_auc/esmfold2_auc_dhd_leaderboard.csv",
    },
}

# copied from py_rosetta/scripts/analysis/pairing_utils.py
ROSETTA_LOWER_IS_BETTER = [
    "binder_score", "surface_hydrophobicity", "interface_dG",
    "interface_dG_SASA_ratio", "interface_delta_unsat_hbonds",
    "interface_delta_unsat_hbonds_percentage",
]
ROSETTA_UNWANTED = ["protein_pair", "cognate_status"]


def rosetta_pooled_best_metric(variant_project, dataset):
    """Concatenate the variant's 4 spm files for this dataset, then return the single
    metric with the highest AUC on the pooled rows."""
    frames = [
        pd.read_csv(ROSETTA_METRICS_DIR / f"{variant_project}_{spm}_{dataset}_combined.csv")
        for spm in SPMS
    ]
    pooled = pd.concat(frames, ignore_index=True)

    best_metric, best_auc = None, -np.inf
    for metric in pooled.columns:
        if metric.strip() in ROSETTA_UNWANTED:
            continue
        oriented = -pooled[metric] if metric in ROSETTA_LOWER_IS_BETTER else pooled[metric]
        mask = pooled[metric].notna()
        if mask.sum() < 2 or pooled.loc[mask, "cognate_status"].nunique() < 2:
            continue
        auc = roc_auc_score(pooled.loc[mask, "cognate_status"], oriented[mask])
        if auc > best_auc:
            best_metric, best_auc = metric, auc
    return best_metric, best_auc


all_datasets_rows = []
for dataset in DATASET_ORDER:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    best = []
    for model in ["AF3", "Boltz-2", "Chai", "ESMFold2"]:
        df = pd.read_csv(LEADERBOARDS[model][dataset])
        top_row = df.loc[df["auc"].idxmax()]
        best.append({"model": model, "best_metric": top_row["metric"], "auc": top_row["auc"]})
    for model, variant_project in ROSETTA_VARIANTS.items():
        best_metric, auc = rosetta_pooled_best_metric(variant_project, dataset)
        best.append({"model": model, "best_metric": best_metric, "auc": auc})

    best_df = pd.DataFrame(best)
    best_df["model"] = pd.Categorical(best_df["model"], categories=MODEL_ORDER, ordered=True)
    best_df = best_df.sort_values("model").reset_index(drop=True)
    best_df.to_csv(results_dir / "best_metric_comparison_with_rosetta.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        best_df["model"].astype(str), best_df["auc"],
        color=[MODEL_COLORS[m] for m in best_df["model"]],
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(f"Best metric per model (incl. Rosetta, pooled) — {dataset}")
    for bar, metric, auc in zip(bars, best_df["best_metric"], best_df["auc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{metric}\n{auc:.3f}", ha="center", va="bottom", fontsize=7.5,
        )
    plt.tight_layout()
    plt.savefig(results_dir / "best_metric_comparison_with_rosetta.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(best_df.to_string(index=False))

    labeled = best_df.copy()
    labeled.insert(0, "dataset", DATASET_LABELS[dataset])
    all_datasets_rows.append(labeled)

# --- combined-both-datasets grouped bar chart ---
combined = pd.concat(all_datasets_rows, ignore_index=True)
combined.to_csv(RESULTS_DIR / "best_metric_all_datasets_with_rosetta.csv", index=False)

fig, ax = plt.subplots(figsize=(11, 6))
n_models = len(MODEL_ORDER)
bar_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))

for i, model in enumerate(MODEL_ORDER):
    offset = (i - (n_models - 1) / 2) * bar_width
    heights, labels = [], []
    for dataset in DATASET_ORDER:
        row = combined[(combined["dataset"] == DATASET_LABELS[dataset]) & (combined["model"] == model)].iloc[0]
        heights.append(row["auc"])
        labels.append(f"{row['best_metric']}\n{row['auc']:.3f}")
    bars = ax.bar(group_positions + offset, heights, width=bar_width * 0.9,
                   color=MODEL_COLORS[model], label=model)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=6)

ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_title("Best metric per model, incl. Rosetta (pooled) — by dataset")
ax.legend(title="Model", loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_metric_all_datasets_with_rosetta.png", dpi=300)
plt.close()

print("\n--- combined ---")
print(combined.to_string(index=False))
