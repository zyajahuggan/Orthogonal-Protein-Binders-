"""Compares PCA results across AF3, Boltz-2, Chai, and ESMFold2, per dataset. Unlike
compare_models.py's AUC comparison, PCA can't be compared metric-by-metric across models
since each model's metric set differs (e.g. AF3 has 14 metrics, ESMFold2 has 7) -- there's
no shared eigendecomposition, and a single metric's raw loading coefficient isn't
comparable across models with different metric-set sizes anyway. Instead this compares two
things that *are* directly comparable: (1) how much variance PC1 (the dominant axis of
variation) explains, and (2) how well PC1 alone separates cognate from non-cognate pairs,
measured as AUC -- i.e. is the metric space's single biggest source of variation also the
thing that tells cognate from non-cognate apart, or is it some other axis entirely (that's
what LDA, in lda_comparison.py, is for -- it explicitly optimizes for separation instead).

Neither pca.py's pca_variance.csv nor pca_loadings.csv include per-sample PC1 scores, so
this recomputes PCA the same way each project's own pca.py does (standardize all metric
columns, eigendecompose the covariance matrix), generalized over each model's own
UNWANTED/LOWER_IS_BETTER columns -- same approach as lda_comparison.py.

PCA eigenvector sign is arbitrary -- np.linalg.eigh can return either sign for a given
component, independently per model -- so PC1 AUC is reported as max(auc, 1 - auc): since
AUC's whole job here is "how separable," not "which direction," the sign doesn't matter.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"

MODEL_ORDER = ["AF3", "Boltz-2", "Chai", "ESMFold2"]
MODEL_COLORS = {"AF3": "#2a78d6", "Boltz-2": "#1baf7a", "Chai": "#eda100", "ESMFold2": "#008300"}

# (metrics path, ID column, separator) per project per dataset -- copied from
# lda_comparison.py, which already established these per-project quirks
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
        cov_matrix = np.cov(X, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        pc1_variance = eigenvalues[0] / eigenvalues.sum()
        pc1_scores = X @ eigenvectors[:, 0]

        y = df["cognate_interaction"].values
        raw_auc = roc_auc_score(y, pc1_scores)
        pc1_auc = max(raw_auc, 1 - raw_auc)  # sign-invariant separation strength

        rows.append({"model": model, "pc1_variance_explained": pc1_variance, "pc1_auc": pc1_auc})

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(results_dir / "pca_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    bars = ax.bar(comparison_df["model"], comparison_df["pc1_variance_explained"],
                   color=[MODEL_COLORS[m] for m in comparison_df["model"]])
    ax.set_ylabel("Fraction of variance explained")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"PC1 variance explained — {dataset}")
    for bar, val in zip(bars, comparison_df["pc1_variance_explained"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=8)

    ax = axes[1]
    bars = ax.bar(comparison_df["model"], comparison_df["pc1_auc"],
                   color=[MODEL_COLORS[m] for m in comparison_df["model"]])
    ax.axhline(0.5, linestyle="--", color="gray", linewidth=0.8, label="Chance (AUC=0.5)")
    ax.set_ylabel("AUC (PC1 score vs. cognate/non-cognate)")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"PC1 as cognate/non-cognate separator — {dataset}")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, comparison_df["pc1_auc"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / "pca_comparison.png", dpi=300)
    plt.close()

    print(f"--- {dataset} ---")
    print(comparison_df.to_string(index=False))
