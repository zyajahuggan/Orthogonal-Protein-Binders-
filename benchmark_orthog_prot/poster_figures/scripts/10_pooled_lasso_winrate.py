"""Poster figure: pooled LASSO's both-cognates-win-rate, in-sample vs. honest
nested-CV, combined into one figure for DHD and COP, with the same "random (1/6)"
reference line the rest of this poster's win-rate figures use.

The nested-CV (honest) win rate is read straight from
cross_model_analysis/results/lasso_pooled_winrate/pooled_{dataset}_lasso_winrate.csv --
that script's leave-one-out nested CV is expensive, so it isn't rerun here.
lasso_pooled_winrate.py must be run first.

The in-sample win rate is newly computed here (cheap: one LogisticRegressionCV fit on
the whole dataset, not a leave-one-out loop) using the same pooled-metrics loading
logic duplicated from lasso_pooled_winrate.py, and checked against the exact same
combos (pairs of cognate protein-pairs with cross non-cognate rows) that script found,
so the two bars are comparing the same set of match-ups.

New file -- does not modify or overwrite lasso_pooled_winrate.py or its outputs.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE / "cross_model_analysis" / "results" / "lasso_pooled_winrate"
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 19,
    "axes.titlesize": 25,
    "axes.titleweight": "bold",
    "axes.labelsize": 23,
    "xtick.labelsize": 20,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

SPMS = ["af3", "boltz", "chai", "esmfold2"]
DATASETS = ["dhd", "cop"]
DATASET_LABELS = {"dhd": "DHD", "cop": "COP"}
INNER_CV_SPLITS = 3

PHYSICS_LOWER_IS_BETTER = [
    "binder_score", "surface_hydrophobicity", "interface_dG",
    "interface_dG_SASA_ratio", "interface_delta_unsat_hbonds",
    "interface_delta_unsat_hbonds_percentage",
]
PHYSICS_UNWANTED = ["protein_pair", "cognate_status"]

IPSAE_BOOKKEEPING = ["ipsae_n0res", "ipsae_n0chn", "ipsae_n0dom", "ipsae_d0res", "ipsae_d0chn",
                      "ipsae_d0dom", "ipsae_nres1", "ipsae_nres2", "ipsae_dist1", "ipsae_dist2"]

NATIVE_CONFIG = {
    "af3": {
        "id_column": "sample",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["sample", "best_seed", "best_sample", "cognate_interaction",
                              "has_clash", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0",
                                     "fraction_disordered"],
        "dhd": ("dhd_metrics_af3_filtered.csv", "_vs_"),
        "cop": ("cop_metrics_af3.csv", "_vs_"),
    },
    "boltz": {
        "id_column": "job_name",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["complex_pde", "complex_ipde"],
        "dhd": ("boltz_metrics_dhd_filtered.csv", "_vs_"),
        "cop": ("boltz_metrics_cop.csv", "_vs_"),
    },
    "chai": {
        "id_column": "project",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["project", "cognate_interaction",
                              "chain_chain_clashes_0_0", "chain_chain_clashes_0_1",
                              "chain_chain_clashes_1_0", "chain_chain_clashes_1_1",
                              "has_inter_chain_clashes", "interface_hbonds"],
        "lower_is_better_columns": [],
        "dhd": ("dhd_metrics_filtered.csv", "_vs_"),
        "cop": ("cop_metrics.csv", "_vs_"),
    },
    "esmfold2": {
        "id_column": "job_name",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"] + IPSAE_BOOKKEEPING,
        "lower_is_better_columns": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
        "dhd": ("dhd_combined_metrics_filtered.csv", "__"),
        "cop": ("cop_combined_metrics.csv", "_vs_"),
    },
}


def _load_native(spm, dataset):
    config = NATIVE_CONFIG[spm]
    filename, sep = config[dataset]
    df = pd.read_csv(BASE / spm / "metrics" / filename)
    metric_columns = [c for c in df.columns if c not in config["unwanted_columns"]]

    data = df[metric_columns].copy()
    for col in config["lower_is_better_columns"]:
        if col in data.columns:
            data[col] = -data[col]
    data.columns = [f"{spm}_native_{c}" for c in data.columns]
    data["_key"] = df[config["id_column"]].apply(lambda s: frozenset(p.lower() for p in s.split(sep)))
    data[f"_label_{spm}_native"] = df[config["cognate_column"]]
    return data


def _load_physics(spm, dataset):
    path = BASE / "py_rosetta" / "metrics" / "metrics" / f"fast_relax_riam_{spm}_{dataset}_combined.csv"
    df = pd.read_csv(path)
    metric_columns = [c for c in df.columns if c not in PHYSICS_UNWANTED]

    data = df[metric_columns].copy()
    for col in PHYSICS_LOWER_IS_BETTER:
        if col in data.columns:
            data[col] = -data[col]
    data.columns = [f"{spm}_physics_{c}" for c in data.columns]
    data["_key"] = df["protein_pair"].apply(lambda s: frozenset(p.lower() for p in s.split("_vs_")))
    data[f"_label_{spm}_physics"] = df["cognate_status"]
    if spm == SPMS[0]:
        data["protein_pair"] = df["protein_pair"]
    return data


def load_pooled_metrics(dataset):
    sources = []
    for spm in SPMS:
        sources.append(_load_native(spm, dataset))
        sources.append(_load_physics(spm, dataset))

    merged = sources[0]
    for src in sources[1:]:
        merged = merged.merge(src, on="_key", how="inner")

    label_columns = [c for c in merged.columns if c.startswith("_label_")]
    merged["cognate_status"] = merged[label_columns[0]]

    feature_columns = [
        c for c in merged.columns
        if any(c.startswith(f"{spm}_native_") or c.startswith(f"{spm}_physics_") for spm in SPMS)
    ]
    result = merged[["protein_pair", "cognate_status"] + feature_columns].copy()
    result.index = result["protein_pair"]
    return result


def find_combos(protein_pairs, cognate_status, sep="_vs_"):
    cognate_pairs = [p for p, c in zip(protein_pairs, cognate_status) if c == 1]
    chain_sets = {p: set(p.split(sep)) for p in cognate_pairs}

    combos = []
    for i, j in itertools.combinations(cognate_pairs, 2):
        chains_i, chains_j = chain_sets[i], chain_sets[j]
        cross_rows = [
            p for p in protein_pairs
            if p not in (i, j)
            and len(set(p.split(sep)) & chains_i) == 1
            and len(set(p.split(sep)) & chains_j) == 1
        ]
        if cross_rows:
            combos.append((i, j, cross_rows))
    return combos


def win_rate_from_scores(scores, combos):
    n_win = sum(
        min(scores[i], scores[j]) > max(scores[c] for c in cross_rows)
        for i, j, cross_rows in combos
    )
    return n_win, len(combos)


in_sample_rates, honest_rates, labels = {}, {}, {}
for dataset in DATASETS:
    merged = load_pooled_metrics(dataset)
    feature_columns = [c for c in merged.columns if c not in ("protein_pair", "cognate_status")]
    data = merged[feature_columns].copy()

    columns_with_nan = data.columns[data.isna().any()].tolist()
    if columns_with_nan:
        data = data.drop(columns=columns_with_nan)

    X = StandardScaler().fit_transform(data.values)
    y = merged["cognate_status"].values

    inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=0)
    lasso_full = LogisticRegressionCV(
        Cs=10, cv=inner_cv, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000,
    )
    lasso_full.fit(X, y)
    in_sample_scores = pd.Series(lasso_full.decision_function(X), index=merged["protein_pair"])

    combos = find_combos(merged["protein_pair"].tolist(), merged["cognate_status"].tolist())
    n_win, n_combos = win_rate_from_scores(in_sample_scores, combos)
    in_sample_rates[dataset] = n_win / n_combos

    honest_df = pd.read_csv(RESULTS_DIR / f"pooled_{dataset}_lasso_winrate.csv")
    honest_n_win = int(honest_df["win"].sum())
    honest_n_combos = len(honest_df)
    honest_rates[dataset] = honest_n_win / honest_n_combos

    labels[dataset] = (f"{n_win}/{n_combos}", f"{honest_n_win}/{honest_n_combos}")
    print(f"{dataset}: in-sample {n_win}/{n_combos}={in_sample_rates[dataset]:.3f}  "
          f"nested-CV {honest_n_win}/{honest_n_combos}={honest_rates[dataset]:.3f}")

# --- grouped bar chart: one group per dataset, in-sample vs. nested-CV within each ---
IN_SAMPLE_COLOR = "#2a78d6"
HONEST_COLOR = "#e34948"

x = np.arange(len(DATASETS))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 8))

in_sample_heights = [in_sample_rates[d] for d in DATASETS]
honest_heights = [honest_rates[d] for d in DATASETS]

bars_in = ax.bar(x - width / 2, in_sample_heights, width, color=IN_SAMPLE_COLOR, label="In-Sample")
bars_honest = ax.bar(x + width / 2, honest_heights, width, color=HONEST_COLOR, label="Nested CV")

for xi, dataset in zip(x, DATASETS):
    in_label, honest_label = labels[dataset]
    ax.text(xi - width / 2, in_sample_rates[dataset] + 0.02, in_label, ha="center", va="bottom", fontsize=17)
    ax.text(xi + width / 2, honest_rates[dataset] + 0.02, honest_label, ha="center", va="bottom", fontsize=17)

ax.axhline(1 / 6, linestyle="--", color="gray", linewidth=1.5, label="Random (1/6)")
ax.set_xticks(x)
ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
ax.set_ylabel("Win Rate")
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Pooled LASSO Win Rate, In-Sample vs. Nested CV")
ax.legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "pooled_lasso_winrate.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"saved {OUT_DIR / 'pooled_lasso_winrate.png'}")
