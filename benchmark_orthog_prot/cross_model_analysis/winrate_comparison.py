"""Compares the both-cognates-win rate for the metrics every model computes under the same
name (iptm, ptm) across AF3, Boltz-2, Chai, and ESMFold2, per dataset --
the win-rate analogue of compare_models.py's AUC comparison. Grouped bar chart: one group
per metric, one bar per model, same visual style as plot_model_comparison.py's
model_auc_comparison plot.

"Both-cognates-win rate" is each project's own pair_stats.py logic, generalized over each
model's own ID column and separator: for every pair of mutually orthogonal cognate pairs
(i.e. the data has cross rows between their two proteins), check whether the worse-scoring
cognate pair still beats the better-scoring cross non-cognate. pair_stats.py only saves a
barplot per model (no CSV), so this recomputes it the same way, using the same PATHS dict as
compare_models.py -- no bootstrap significance testing here, unlike compare_models.py,
since pair_stats.py's own analysis doesn't do one either and the combination counts are
already small (14-15 per dataset) before any resampling.

Rosetta and ProteinMPNN aren't included, for the same reason as the other cross-model
comparisons: neither has metrics in the per-sample CSV format this logic expects.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

# metrics every project computes under the same name and, per compare_models.py, all orient
# the same way (none are lower-is-better), so no per-model sign-flipping is needed here
SHARED_METRICS = ["iptm", "ptm"]

# (metrics path, ID column, separator) per project per dataset -- copied from
# compare_models.py, which already established these per-project quirks
PATHS = {
    "AF3": {
        "cop": (BASE / "af3/metrics/cop_metrics_af3.csv", "sample", "_vs_"),
        "dhd": (BASE / "af3/metrics/dhd_metrics_af3_filtered.csv", "sample", "_vs_"),
    },
    "Boltz-2": {
        "cop": (BASE / "boltz/metrics/boltz_metrics_cop.csv", "job_name", "_vs_"),
        "dhd": (BASE / "boltz/metrics/boltz_metrics_dhd_filtered.csv", "job_name", "_vs_"),
    },
    "Chai": {
        "cop": (BASE / "chai/metrics/cop_metrics.csv", "project", "_vs_"),
        "dhd": (BASE / "chai/metrics/dhd_metrics_filtered.csv", "project", "_vs_"),
    },
    "ESMFold2": {
        "cop": (BASE / "esmfold2/metrics/cop_combined_metrics.csv", "job_name", "_vs_"),
        "dhd": (BASE / "esmfold2/metrics/dhd_combined_metrics_filtered.csv", "job_name", "__"),
    },
}


def find_combos(df, id_col, sep):
    """Every pair of mutually orthogonal cognate pairs that have cross non-cognate rows
    between them, plus those cross row indices -- same discovery logic as pair_stats.py,
    generalized over id_col instead of hardcoding it."""
    cognate_indices = df.index[df["cognate_interaction"] == 1].tolist()
    chain_sets = {i: set(df.loc[i, id_col].split(sep)) for i in cognate_indices}

    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i, chains_j = chain_sets[i], chain_sets[j]
        cross_rows = [
            idx for idx in df.index
            if idx not in (i, j)
            and len(set(df.loc[idx, id_col].split(sep)) & chains_i) == 1
            and len(set(df.loc[idx, id_col].split(sep)) & chains_j) == 1
        ]
        if cross_rows:
            combos.append((i, j, cross_rows))
    return combos


for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)
        combos = find_combos(df, id_col, sep)

        for metric in SHARED_METRICS:
            scores = df[metric]
            n_win = 0
            for i, j, cross_rows in combos:
                worst_cognate = min(scores.loc[i], scores.loc[j])
                best_noncognate = max(scores.loc[idx] for idx in cross_rows)
                n_win += worst_cognate > best_noncognate
            rows.append({
                "model": model, "metric": metric,
                "win_rate": n_win / len(combos), "n_win": n_win, "n_combos": len(combos),
            })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(results_dir / "winrate_comparison.csv", index=False)

    x = np.arange(len(SHARED_METRICS))
    width = 0.8 / len(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, model in enumerate(MODEL_ORDER):
        rates = [comparison_df.query("model == @model and metric == @m")["win_rate"].item() for m in SHARED_METRICS]
        ax.bar(x + i * width, rates, width, label=model, color=MODEL_COLORS[model])
    ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=0.8, label="random (1/6)")
    ax.set_xticks(x + width * (len(MODEL_ORDER) - 1) / 2)
    ax.set_xticklabels(SHARED_METRICS)
    ax.set_ylabel("Both-cognates-win rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Both-cognates-win rate by model and metric — {dataset}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "winrate_comparison.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(comparison_df.to_string(index=False))
