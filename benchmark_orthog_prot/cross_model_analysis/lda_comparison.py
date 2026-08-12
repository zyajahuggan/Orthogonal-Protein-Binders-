"""Compares each model's LDA performance -- in-sample AUC (optimistic, fit and scored on
the same data) vs. leave-one-out cross-validated AUC (the generalization number) -- across
AF3, Boltz-2, Chai, and ESMFold2, per dataset.

Each project's lda.py only saves the per-sample scores CSV (lda_score, cv_lda_score), not
the summary AUCs themselves -- those are only printed to stdout. So, same reasoning as
best_rank_comparison.py and best_margin_comparison.py, this recomputes LDA the same way
each project's own lda.py does (standardize all metric columns, fit LDA, score in-sample,
then leave-one-out cross-validate), generalized over each model's own UNWANTED/
LOWER_IS_BETTER columns rather than importing per-project pairing_utils.py.

Rosetta and ProteinMPNN aren't included, for the same reason as the other cross-model
comparisons: neither has metrics in the per-sample CSV format this expects.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# best_rank_comparison.py, which already established these per-project quirks
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

# copied from each project's own pairing_utils.py
LOWER_IS_BETTER = {
    "AF3": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0", "fraction_disordered"],
    "Boltz-2": ["complex_pde", "complex_ipde"],
    "Chai": [],
    "ESMFold2": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
}
# raw bookkeeping counts/normalization constants add_ipsae_metrics.py adds alongside the
# real ipSAE scores -- not confidence signals themselves, so excluded from the feature set
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

for dataset in ["cop", "dhd"]:
    results_dir = RESULTS_DIR / dataset
    results_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODEL_ORDER:
        path, id_col, sep = PATHS[model][dataset]
        df = pd.read_csv(path)

        metric_columns = [c for c in df.columns if c not in UNWANTED[model]]
        data = df[metric_columns].copy()
        for col in LOWER_IS_BETTER[model]:
            if col in data.columns:
                data[col] = -data[col]

        X = StandardScaler().fit_transform(data.values)
        y = df["cognate_interaction"].values

        lda = LinearDiscriminantAnalysis()
        lda_scores = lda.fit_transform(X, y).ravel()
        in_sample_auc = roc_auc_score(y, lda_scores)

        cv = LeaveOneOut()
        cv_scores = cross_val_predict(lda, X, y, cv=cv, method="decision_function")
        cv_auc = roc_auc_score(y, cv_scores)

        rows.append({
            "model": model, "n_metrics": len(metric_columns),
            "in_sample_auc": in_sample_auc, "cv_auc": cv_auc,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(results_dir / "lda_comparison.csv", index=False)

    x = np.arange(len(MODEL_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, comparison_df["in_sample_auc"], width, label="In-sample", color="steelblue")
    ax.bar(x + width / 2, comparison_df["cv_auc"], width, label="Cross-validated", color="firebrick")
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="Chance (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model"])
    ax.set_ylabel("LDA AUC")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"LDA: In-sample vs. Cross-validated AUC by model — {dataset}")
    ax.legend(fontsize=8)

    for xi, row in zip(x, comparison_df.itertuples()):
        ax.text(xi - width / 2, row.in_sample_auc + 0.02, f"{row.in_sample_auc:.2f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, row.cv_auc + 0.02, f"{row.cv_auc:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / "lda_comparison.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(comparison_df.to_string(index=False))
