"""Both-cognates-win-rate comparison extended with Rosetta's two scoring variants (RIAM,
FastRelax), alongside best_winrate_all_datasets.py's AF3/Boltz-2/Chai/ESMFold2 bars. New
file -- does not modify or overwrite best_winrate_all_datasets.py or its outputs.

Rosetta re-scores each DL model's predicted structures rather than generating its own, so
there's one riam_{spm}_{dataset}_combined.csv and one fast_relax_riam_{spm}_{dataset}_combined.csv
per spm (af3/boltz/chai/esmfold2). Unlike the AUC comparison (best_metric_comparison_with_rosetta.py),
win-rate can't just concatenate the 4 spm files: the "both-cognates-win" combos require
matching cognate row IDs against competitor row IDs sharing a protein name, and each spm
file uses its own protein-ID casing (af3 lowercases "dhd150a_vs_dhd150a", boltz uses
"DHD150a_vs_DHD150a") -- pooling raw rows would only find matches within a spm's own rows
anyway (case differs across spms) but does so silently, so it's safer to compute wins/combos
per spm explicitly and sum them: n_win and n_combos are each summed across the 4 spms per
candidate metric, then win_rate = total_n_win / total_n_combos, and the metric with the
highest pooled win_rate is picked as RIAM's/FastRelax's "best metric" -- same selection
rule best_winrate_all_datasets.py uses per DL model, just summed over 4 structure sets
before picking instead of 1.

ROSETTA_LOWER_IS_BETTER/ROSETTA_UNWANTED are copied from py_rosetta/scripts/analysis/
pairing_utils.py, same cross-project duplication precedent as py_rosetta/scripts/analysis/
riam_vs_own_metric_comparison.py. find_combos is best_winrate_all_datasets.py's own
combo-discovery logic, generalized over cognate_col since Rosetta's files use
"cognate_status" instead of "cognate_interaction".
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
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

# copied from py_rosetta/scripts/analysis/pairing_utils.py
ROSETTA_LOWER_IS_BETTER = [
    "binder_score", "surface_hydrophobicity", "interface_dG",
    "interface_dG_SASA_ratio", "interface_delta_unsat_hbonds",
    "interface_delta_unsat_hbonds_percentage",
]
ROSETTA_UNWANTED = ["protein_pair", "cognate_status"]


def orient_scores(df, metric, lower_is_better_columns):
    return -df[metric] if metric in lower_is_better_columns else df[metric]


def find_combos(df, id_col, sep, cognate_col):
    """Every pair of mutually orthogonal cognate pairs that have cross non-cognate rows
    between them, plus those cross row indices -- same discovery logic as pair_stats.py,
    generalized over cognate_col since Rosetta's files use "cognate_status" instead of
    "cognate_interaction"."""
    cognate_indices = df.index[df[cognate_col] == 1].tolist()
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
    """Every metric's both-cognates-win rate for this model/dataset, then the single
    best one by win rate -- ties keep whichever metric comes first in the CSV's column
    order, same as pair_stats.py's sort_values."""
    path, id_col, sep = PATHS[model][dataset]
    df = pd.read_csv(path)
    combos = find_combos(df, id_col, sep, "cognate_interaction")

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


def rosetta_pooled_best_winrate(variant_project, dataset):
    """Per spm, compute each candidate metric's (n_win, n_combos); sum both across the
    4 spms per metric; pick the metric with the highest pooled win_rate."""
    per_spm = []
    for spm in SPMS:
        path = ROSETTA_METRICS_DIR / f"{variant_project}_{spm}_{dataset}_combined.csv"
        df = pd.read_csv(path)
        combos = find_combos(df, "protein_pair", "_vs_", "cognate_status")
        per_spm.append((df, combos))

    totals = {}
    for metric in per_spm[0][0].columns:
        if metric.strip() in ROSETTA_UNWANTED:
            continue
        total_win, total_combos = 0, 0
        for df, combos in per_spm:
            oriented = orient_scores(df, metric, ROSETTA_LOWER_IS_BETTER)
            total_win += sum(
                min(oriented.loc[i], oriented.loc[j]) > max(oriented.loc[idx] for idx in cross_rows)
                for i, j, cross_rows in combos
            )
            total_combos += len(combos)
        totals[metric] = (total_win, total_combos)

    best_metric = max(totals, key=lambda m: totals[m][0] / totals[m][1])
    n_win, n_combos = totals[best_metric]
    return best_metric, n_win, n_combos


# --- combine both datasets into one table ---
rows = []
for dataset in DATASET_ORDER:
    for model in ["AF3", "Boltz-2", "Chai", "ESMFold2"]:
        best_metric, n_win, n_combos = best_winrate_metric(model, dataset)
        rows.append({
            "dataset": DATASET_LABELS[dataset], "model": model, "best_metric": best_metric,
            "win_rate": n_win / n_combos, "n_win": n_win, "n_combos": n_combos,
        })
    for model, variant_project in ROSETTA_VARIANTS.items():
        best_metric, n_win, n_combos = rosetta_pooled_best_winrate(variant_project, dataset)
        rows.append({
            "dataset": DATASET_LABELS[dataset], "model": model, "best_metric": best_metric,
            "win_rate": n_win / n_combos, "n_win": n_win, "n_combos": n_combos,
        })

combined = pd.DataFrame(rows)
combined.to_csv(RESULTS_DIR / "best_winrate_all_datasets_with_rosetta.csv", index=False)

# --- grouped bar chart: one group per dataset, one bar per model within the group ---
fig, ax = plt.subplots(figsize=(11, 6))
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
                label, ha="center", va="bottom", fontsize=6)

ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=0.8, label="random (1/6)")
ax.set_ylabel("Both-cognates-win rate")
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(group_positions)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
ax.set_title("Best both-cognates-win rate per model, incl. Rosetta (pooled), by dataset")
ax.legend(title="Model", loc="lower right", fontsize=8)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_winrate_all_datasets_with_rosetta.png", dpi=300)
plt.close()

print(combined.to_string(index=False))
