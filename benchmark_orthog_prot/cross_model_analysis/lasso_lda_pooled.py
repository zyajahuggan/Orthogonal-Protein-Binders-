"""One row per protein pair, per dataset (dhd, cop): all 4 SPMs' (AF3,
Boltz-2, Chai, ESMFold2) own confidence metrics plus fast_relax_riam physics metrics
computed on each SPM's own relaxed structure, pooled into a single feature matrix
(~100+ columns). Fits LDA and L1-regularized (LASSO) logistic regression on that
pooled matrix and compares in-sample vs. cross-validated AUC for both, following the
same honesty standard as py_rosetta/scripts/analysis/lda_combined.py and
cross_model_analysis/lda_overfitting_headline.py: an optimistic in-sample number next
to the number that actually estimates generalization.

Only fast_relax_riam is included as the physics source (not riam), per request.

Unlike plain LDA, LASSO logistic regression's L1 penalty can drive a coefficient to
exactly 0 -- i.e. it can genuinely ignore a metric, not just downweight it. Whether
that produces a smaller, still-useful feature set (rather than just a second flavor of
overfitting) is exactly what the AUC comparison below is checking.

DHD has only 6 cognate pairs, too few for leave-one-out AUC to mean much on its own --
reported anyway for consistency with the rest of the codebase, but flagged in the
printed output rather than presented as a reliable number.

For LASSO, picking the regularization strength C by cross-validating on the same data
the final AUC is reported on would repeat the exact leakage this analysis is trying to
avoid. So "cross-validated" LASSO AUC here is a nested CV: for each outer
leave-one-out fold, an inner StratifiedKFold picks C using only that fold's training
data (via LogisticRegressionCV), then the resulting model scores the single held-out
point. "In-sample" LASSO AUC instead fits LogisticRegressionCV once on all the data
(same C-selection process, but the final refit sees everything) and scores on that
same data -- the same kind of optimistic number LDA's in-sample AUC already is,
included so the two classifiers are compared on equal footing.

Merge strategy mirrors py_rosetta/scripts/analysis/lda_combined.py's
load_combined_metrics: match rows by frozenset(protein_pair.split(sep)) rather than
raw id string, since each source names/separates its id column differently. Extended
here to inner-join across all 4 SPMs' native metrics and fast_relax_riam physics
metrics (8 sources total), instead of one SPM's native + physics only. NATIVE_CONFIG
is duplicated (not imported) from that file, matching this codebase's own precedent
(riam_vs_own_metric_comparison.py) for crossing between project directories.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results" / "lasso_lda_pooled"
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
    # lowercased: AF3 names proteins lowercase (dhd37_bbbba) while Boltz/Chai/ESMFold2
    # use mixed case (DHD37_ABXBb) for the same protein, so raw-case keys wouldn't match
    # across SPMs even though lda_combined.py's single-SPM merge never hit this
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
    n_mismatch = (merged[label_columns].nunique(axis=1) > 1).sum()
    if n_mismatch:
        print(f"WARNING: {dataset}: {n_mismatch} rows disagree on cognate label across sources")
    merged["cognate_status"] = merged[label_columns[0]]

    feature_columns = [
        c for c in merged.columns
        if any(c.startswith(f"{spm}_native_") or c.startswith(f"{spm}_physics_") for spm in SPMS)
    ]
    result = merged[["protein_pair", "cognate_status"] + feature_columns].copy()
    result.index = result["protein_pair"]
    return result


def nested_lasso_auc(X, y):
    """Honest LASSO AUC: for each leave-one-out fold, C is picked by an inner CV that
    only sees that fold's training data, so the held-out point never influences the
    choice of C used to score it."""
    outer = LeaveOneOut()
    preds = np.zeros(len(y))
    for train_idx, test_idx in outer.split(X):
        inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=0)
        clf = LogisticRegressionCV(
            Cs=10, cv=inner_cv, penalty="l1", solver="liblinear",
            scoring="roc_auc", max_iter=5000,
        )
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.decision_function(X[test_idx])
    return roc_auc_score(y, preds)


for dataset in DATASETS:
    label = f"pooled_{dataset}"
    merged = load_pooled_metrics(dataset)
    feature_columns = [c for c in merged.columns if c not in ("protein_pair", "cognate_status")]
    data = merged[feature_columns].copy()

    columns_with_nan = data.columns[data.isna().any()].tolist()
    if columns_with_nan:
        print(f"{label}: dropping metrics with missing data: {columns_with_nan}")
        data = data.drop(columns=columns_with_nan)

    X = StandardScaler().fit_transform(data.values)
    y = merged["cognate_status"].values

    print(f"\n--- {label} ---")
    print(f"n={len(y)}, cognate={int(y.sum())}, non-cognate={int(len(y) - y.sum())}, n_metrics={data.shape[1]}")
    if dataset == "dhd":
        print("CAUTION: only 6 cognate pairs -- cross-validated numbers below are unreliable, "
              "reported for consistency only.")

    # --- LDA ---
    lda = LinearDiscriminantAnalysis()
    lda_in_sample_scores = lda.fit_transform(X, y).ravel()
    lda_in_sample_auc = roc_auc_score(y, lda_in_sample_scores)
    lda_cv_scores = cross_val_predict(lda, X, y, cv=LeaveOneOut(), method="decision_function")
    lda_cv_auc = roc_auc_score(y, lda_cv_scores)
    print(f"LDA:   in-sample AUC={lda_in_sample_auc:.3f}   cross-validated AUC={lda_cv_auc:.3f}")

    # --- LASSO logistic regression ---
    inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=0)
    lasso_full = LogisticRegressionCV(
        Cs=10, cv=inner_cv, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000,
    )
    lasso_full.fit(X, y)
    lasso_in_sample_scores = lasso_full.decision_function(X)
    lasso_in_sample_auc = roc_auc_score(y, lasso_in_sample_scores)
    lasso_nested_auc = nested_lasso_auc(X, y)
    n_nonzero = int(np.count_nonzero(lasso_full.coef_))
    print(f"LASSO: in-sample AUC={lasso_in_sample_auc:.3f}   nested cross-validated AUC={lasso_nested_auc:.3f}   "
          f"(C={lasso_full.C_[0]:.4g}, {n_nonzero}/{data.shape[1]} metrics kept nonzero)")

    # --- save scores ---
    scores_df = pd.DataFrame({
        "protein_pair": merged["protein_pair"],
        "cognate_status": y,
        "lda_in_sample": lda_in_sample_scores,
        "lda_cv": lda_cv_scores,
        "lasso_in_sample": lasso_in_sample_scores,
    })
    scores_df.to_csv(RESULTS_DIR / f"{label}_scores.csv", index=False)

    # --- save LASSO coefficients (the actual feature-selection answer) ---
    coef_df = pd.DataFrame({
        "metric": data.columns,
        "coefficient": lasso_full.coef_.ravel(),
    })
    coef_df["nonzero"] = coef_df["coefficient"] != 0
    coef_df = coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index)
    coef_df.to_csv(RESULTS_DIR / f"{label}_lasso_coefficients.csv", index=False)

    # --- bar chart: in-sample vs. cross-validated AUC, both classifiers ---
    bar_labels = ["LDA\nin-sample", "LDA\ncross-val", "LASSO\nin-sample", "LASSO\nnested cross-val"]
    bar_values = [lda_in_sample_auc, lda_cv_auc, lasso_in_sample_auc, lasso_nested_auc]
    bar_colors = ["steelblue", "firebrick", "steelblue", "firebrick"]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    bars = ax.bar(bar_labels, bar_values, color=bar_colors)
    ax.axhline(0.5, color="gray", linestyle="--", label="Chance (0.5)")
    for bar, value in zip(bars, bar_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}",
                ha="center", va="bottom")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.08)
    title = f"Pooled LDA vs. LASSO — {DATASET_LABELS[dataset]}"
    if dataset == "dhd":
        title += " (n=6 cognate — CV unreliable)"
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{label}_auc_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --- ROC curve overlay ---
    fpr_lda_in, tpr_lda_in, _ = roc_curve(y, lda_in_sample_scores)
    fpr_lda_cv, tpr_lda_cv, _ = roc_curve(y, lda_cv_scores)
    fpr_lasso_in, tpr_lasso_in, _ = roc_curve(y, lasso_in_sample_scores)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr_lda_in, tpr_lda_in, color="steelblue", linestyle="-", label=f"LDA in-sample (AUC={lda_in_sample_auc:.2f})")
    plt.plot(fpr_lda_cv, tpr_lda_cv, color="firebrick", linestyle="-", label=f"LDA cross-val (AUC={lda_cv_auc:.2f})")
    plt.plot(fpr_lasso_in, tpr_lasso_in, color="steelblue", linestyle="--", label=f"LASSO in-sample (AUC={lasso_in_sample_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle=":", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Pooled LDA vs. LASSO ROC — {DATASET_LABELS[dataset]}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{label}_roc_comparison.png", dpi=300)
    plt.close()

print(f"\nSaved all results to {RESULTS_DIR}")
