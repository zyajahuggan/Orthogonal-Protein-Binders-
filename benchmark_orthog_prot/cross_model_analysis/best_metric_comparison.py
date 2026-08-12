"""Compares each model's single BEST metric (by AUC) against the others' best metrics,
per dataset. Unlike compare_models.py, this does not restrict to metrics shared by name
across models -- each model is represented by whichever metric scored highest for it, even
if that metric doesn't exist for the others (e.g. AF3's chain_pair_pae_min_0_1 vs. Chai's
aggregate_score).

Reads the existing per-model AUC leaderboard CSVs -- no recomputation. Rosetta and
ProteinMPNN aren't included: neither has metrics run through this project's AUC leaderboard
pipeline yet.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# same colors as plot_model_comparison.py, so a model's bar color stays consistent
# across every plot in cross_model_analysis/
MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

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

for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    best = []
    for model in MODEL_ORDER:
        df = pd.read_csv(LEADERBOARDS[model][dataset])
        top_row = df.loc[df["auc"].idxmax()]
        best.append({"model": model, "best_metric": top_row["metric"], "auc": top_row["auc"]})

    best_df = pd.DataFrame(best)
    best_df.to_csv(results_dir / "best_metric_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        best_df["model"], best_df["auc"],
        color=[MODEL_COLORS[m] for m in best_df["model"]],
    )
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8)
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.15)  # headroom above the tallest bar so its label doesn't hit the title
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title(f"Best metric per model — {dataset}")

    # direct-label each bar with its metric name and AUC, since only 4 bars
    for bar, metric, auc in zip(bars, best_df["best_metric"], best_df["auc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{metric}\n{auc:.3f}", ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(results_dir / "best_metric_comparison.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(best_df.to_string(index=False))
