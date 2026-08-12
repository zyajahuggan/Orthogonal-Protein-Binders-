"""Pooled LASSO's per-pair score run through the codebase's existing both-cognates-win-rate
check instead of AUC: for every pair of cognate protein pairs that has cross non-cognate
rows between them (find_combos, same discovery logic as winrate_comparison.py), win if the
worse-scoring cognate pair still beats the best-scoring cross non-cognate on the pooled
LASSO score.

Uses the honest nested-cross-validated LASSO score (each pair scored by a model that never
saw it, with the regularization strength C chosen inside that same holdout) rather than the
in-sample score, for the same reason lasso_lda_pooled.py reports both numbers separately: a
win rate computed on in-sample scores would be inflated by the same overfitting that let
in-sample AUC hit 1.0 there.

DHD has 6 cognate pairs, so all 15 of the "6 choose 2" combos have cross rows between them
-> n=15. COP has 28 cognate pairs, but they're organized as 14 independent groups
of (2 cognate + 2 non-cognate) rows with no cross-group pairing anywhere in the data, so
only the 14 within-group combos have cross rows -- the other ~364 pairwise combinations
across groups get filtered out by find_combos automatically, with no special-casing needed
here for the two datasets' different structure.

Loading logic (NATIVE_CONFIG, _load_native, _load_physics, load_pooled_metrics) is
duplicated, not imported, from lasso_lda_pooled.py -- same precedent that file itself
follows relative to lda_combined.py, and it avoids re-running that script's own top-level
analysis (and overwriting its plots) as a side effect of importing it.
"""

import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lasso_pooled_winrate"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the feature set
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
    """Inner-join all 4 SPMs' native metrics + fast_relax_riam physics metrics (8
    sources) into one row per protein pair, keyed by which two proteins are paired."""
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


def nested_lasso_scores(X, y):
    """Same nested leave-one-out procedure as lasso_lda_pooled.py's nested_lasso_auc, but
    returns the per-point honest scores themselves instead of collapsing them into a single
    AUC -- the win-rate check below needs a score per protein pair, not a summary number."""
    outer = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in outer.split(X):
        inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=0)
        clf = LogisticRegressionCV(
            Cs=10, cv=inner_cv, penalty="l1", solver="liblinear",
            scoring="roc_auc", max_iter=5000, random_state=0,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.decision_function(X[test_idx])
    return preds


def find_combos(protein_pairs, cognate_status, sep="_vs_"):
    """Every pair of cognate protein pairs that has cross non-cognate rows between them --
    same discovery logic as winrate_comparison.py's find_combos, generalized to the pooled
    merged table's single protein_pair/cognate_status columns instead of a per-SPM id
    column with its own separator."""
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


for dataset in DATASETS:
    label = f"pooled_{dataset}"
    merged = load_pooled_metrics(dataset)
    feature_columns = [c for c in merged.columns if c not in ("protein_pair", "cognate_status")]
    data = merged[feature_columns].copy()

    columns_with_nan = data.columns[data.isna().any()].tolist()
    if columns_with_nan:
        data = data.drop(columns=columns_with_nan)

    X = StandardScaler().fit_transform(data.values)
    y = merged["cognate_status"].values
    scores = pd.Series(nested_lasso_scores(X, y), index=merged["protein_pair"])

    combos = find_combos(merged["protein_pair"].tolist(), merged["cognate_status"].tolist())

    rows = []
    for i, j, cross_rows in combos:
        worst_cognate = min(scores[i], scores[j])
        best_noncognate = max(scores[c] for c in cross_rows)
        rows.append({
            "cognate_pair_1": i,
            "cognate_pair_2": j,
            "worst_cognate_score": worst_cognate,
            "best_noncognate_score": best_noncognate,
            "win": worst_cognate > best_noncognate,
        })

    combo_df = pd.DataFrame(rows)
    combo_df.to_csv(RESULTS_DIR / f"{label}_lasso_winrate.csv", index=False)

    n_win = int(combo_df["win"].sum())
    n_combos = len(combo_df)
    nested_auc = roc_auc_score(y, scores.values)
    print(f"--- {label} ---")
    print(f"nested-CV LASSO AUC (same honest score, whole dataset): {nested_auc:.3f}")
    print(f"nested-CV LASSO win rate: {n_win}/{n_combos} = {n_win / n_combos:.3f}")

print(f"\nSaved all results to {RESULTS_DIR}")
