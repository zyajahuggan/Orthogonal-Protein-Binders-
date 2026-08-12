"""Compares each model's single BEST metric (by both-cognates-win rate) against the
others' best metrics, per dataset, then combines both datasets into one table and one
grouped-bar chart. The win-rate analogue of best_metric_all_datasets.py, which does the
same thing for AUC.

Unlike winrate_comparison.py -- which only compares metrics every model computes under
the same name (iptm, ptm) -- each model here is represented by whichever metric scored
the highest win rate for it alone, even if that metric doesn't exist for the others.
Same relationship best_metric_comparison.py has to compare_models.py.

"Both-cognates-win rate" is each project's own pair_stats.py logic (for every pair of
mutually orthogonal cognate pairs, does the worse-scoring cognate pair still beat the
best-scoring cross non-cognate), recomputed here per metric per model since pair_stats.py
only saves a barplot, no CSV. Same PATHS/LOWER_IS_BETTER dicts as best_metric_auc_bootstrap.py
(copied, not imported, to keep this script standalone).
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
DATASET_ORDER = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}

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
LOWER_IS_BETTER = {
    "AF3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "Boltz-2": ["complex_pde", "complex_ipde"],
    "Chai": [],
    "ESMFold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the metric search
IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]
UNWANTED = {
    "AF3": ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash",
            "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Boltz-2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
    "Chai": ["project", "cognate_interaction", "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
             "chain_chain_clashes_1_0", "chain_chain_clashes_1_1", "has_inter_chain_clashes", "interface_hbonds"],
    "ESMFold2": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
}


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def find_combos(df, id_col, sep):
    """Every pair of mutually orthogonal cognate pairs that have cross non-cognate rows
    between them, plus those cross row indices -- same discovery logic as pair_stats.py."""
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


def best_winrate_metric(model, dataset):
    """Every metric's both-cognates-win rate for this model/dataset (pair_stats.py's own
    logic), then the single best one by win rate -- ties keep whichever metric comes
    first in the CSV's column order, same as pair_stats.py's sort_values."""
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    combos = find_combos(df, id_col, sep)

    win_rates = {}
    for metric in df.columns:
        if metric.strip() in UNWANTED[model]:
            continue
        oriented = orient_scores(df, metric, LOWER_IS_BETTER[model])
        n_win = sum(
            min(oriented.loc[i], oriented.loc[j]) > max(oriented.loc[idx] for idx in cross_rows)
            for i, j, cross_rows in combos
        )
        win_rates[metric] = n_win

    best_metric = max(win_rates, key=win_rates.get)
    return best_metric, win_rates[best_metric], len(combos)


# --- combine both datasets into one table ---
rows = []
for dataset in DATASET_ORDER:
    for model in MODEL_ORDER:
        best_metric, n_win, n_combos = best_winrate_metric(model, dataset)
        rows.append({
            "dataset": DATASET_LABELS[dataset], "model": model, "best_metric": best_metric,
            "win_rate": n_win / n_combos, "n_win": n_win, "n_combos": n_combos,
        })

combined = pd.DataFrame(rows)
combined.to_csv(RESULTS_DIR / "best_winrate_all_datasets.csv", index=False)

# --- grouped bar chart: one group per dataset, one bar per model within the group ---
fig, ax = plt.subplots(figsize=(9, 6))
n_models = len(MODEL_ORDER)
bar_width = 0.8 / n_models
group_positions = np.arange(len(DATASET_ORDER))

for i, model in enumerate(MODEL_ORDER):
    offset = (i - (n_models - 1) / 2) * bar_width
    heights, labels = [], []
    for dataset in DATASET_ORDER:
        row = combined[(combined["dataset"] == DATASET_LABELS[dataset]) & (combined["model"] == model)].iloc[0]
        heights.append(row["win_rate"])
        labels.append(f"{row['best_metric']}\n{row['n_win']}/{row['n_combos']}")
    bars = ax.bar(group_positions + offset, heights, width=bar_width * 0.9,
                   color=MODEL_COLORS[model], label=model)
    for bar, label in zip(bars, labels):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                label, ha="center", va="bottom", fontsize=6.5)

ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=0.8, label="random (1/6)")
ax.set_ylabel("Both-cognates-win rate")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_title("Best both-cognates-win rate per model, by dataset")
ax.legend(title="Model", loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_winrate_all_datasets.png", dpi=300)
plt.close()

print(combined.to_string(index=False))
