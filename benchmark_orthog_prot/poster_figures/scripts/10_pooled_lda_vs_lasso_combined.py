"""Poster version of cross_model_analysis/lasso_lda_pooled.py's AUC-comparison bar
chart: pooled LDA vs. LASSO (in-sample and cross-validated AUC), combined into one
figure for DHD and COP instead of lasso_lda_pooled.py's two separate PNGs. Reuses that
script's exact pooling/fitting logic (duplicated, not imported, matching this codebase's
own precedent for cross-directory reuse -- importing would re-trigger its whole
top-level compute-and-plot loop) since it doesn't save the summary AUC numbers to a
CSV, only per-pair scores.

New file -- does not modify or overwrite lasso_lda_pooled.py or its outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent.parent

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "figure.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 15,
    "ytick.labelsize": 16,
    "legend.fontsize": 15,
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


def nested_lasso_auc(X, y):
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


auc_by_dataset = {}
for dataset in DATASETS:
    merged = load_pooled_metrics(dataset)
    feature_columns = [c for c in merged.columns if c not in ("protein_pair", "cognate_status")]
    data = merged[feature_columns].copy()

    columns_with_nan = data.columns[data.isna().any()].tolist()
    if columns_with_nan:
        data = data.drop(columns=columns_with_nan)

    X = StandardScaler().fit_transform(data.values)
    y = merged["cognate_status"].values

    lda = LinearDiscriminantAnalysis()
    lda_in_sample_scores = lda.fit_transform(X, y).ravel()
    lda_in_sample_auc = roc_auc_score(y, lda_in_sample_scores)
    lda_cv_scores = cross_val_predict(lda, X, y, cv=LeaveOneOut(), method="decision_function")
    lda_cv_auc = roc_auc_score(y, lda_cv_scores)

    inner_cv = StratifiedKFold(n_splits=INNER_CV_SPLITS, shuffle=True, random_state=0)
    lasso_full = LogisticRegressionCV(
        Cs=10, cv=inner_cv, penalty="l1", solver="liblinear", scoring="roc_auc", max_iter=5000,
    )
    lasso_full.fit(X, y)
    lasso_in_sample_auc = roc_auc_score(y, lasso_full.decision_function(X))
    lasso_nested_auc = nested_lasso_auc(X, y)

    auc_by_dataset[dataset] = {
        "lda_in_sample": lda_in_sample_auc, "lda_cv": lda_cv_auc,
        "lasso_in_sample": lasso_in_sample_auc, "lasso_nested_cv": lasso_nested_auc,
    }
    print(f"--- {dataset} --- LDA in-sample={lda_in_sample_auc:.3f} LDA CV={lda_cv_auc:.3f} "
          f"LASSO in-sample={lasso_in_sample_auc:.3f} LASSO nested-CV={lasso_nested_auc:.3f}")

# --- grouped bar chart: one group per AUC type, one bar per dataset within the group ---
bar_types = ["lda_in_sample", "lda_cv", "lasso_in_sample", "lasso_nested_cv"]
bar_type_labels = ["LDA\nIn-Sample", "LDA\nCross-Val", "LASSO\nIn-Sample", "LASSO\nNested CV"]
DATASET_COLORS = {"dhd": "#2a78d6", "cop": "#e34948"}

x = np.arange(len(bar_types))
width = 0.35
fig, ax = plt.subplots(figsize=(13, 8.5))

for offset, dataset in zip([-width / 2, width / 2], DATASETS):
    heights = [auc_by_dataset[dataset][bt] for bt in bar_types]
    bars = ax.bar(x + offset, heights, width, color=DATASET_COLORS[dataset], label=DATASET_LABELS[dataset])
    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.2f}", ha="center", va="bottom", fontsize=14)

ax.axhline(0.5, color="gray", linestyle="--", linewidth=1.5, label="Chance (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(bar_type_labels)
ax.set_ylabel("AUC")
ax.set_ylim(0, 1.1)
ax.set_title("Pooled LDA vs. LASSO — DHD and COP\n(DHD: n=6 cognate pairs — cross-validated numbers unreliable)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "pooled_lda_vs_lasso_combined.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"saved {OUT_DIR / 'pooled_lda_vs_lasso_combined.png'}")
