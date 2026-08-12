"""Same question as lda.py -- is there a linear combination of metrics with high AUC
*and* high win rate -- but now the feature set is each spm's own native confidence
metrics (iptm/ptm/plddt/pae/...) combined with the physics-based metrics computed by
fast_relax_riam on that same spm's predicted structures. Answers "does adding the DL
model's own confidence signal on top of the Rosetta physics metrics help."

Native and physics rows are matched by which two proteins are paired, not by raw id
string, since each source names/separates its id column differently (e.g. esmfold2's
native dhd file uses "__" while fast_relax_riam uses "_vs_"). Confirmed 1:1 matched
for every spm/dataset combination before writing this (no unmatched rows either way).

NATIVE_CONFIG mirrors each spm's own pairing_utils.py (id column, cognate column,
excluded columns, lower-is-better columns) -- if those files change, update here too.
"""

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from pairing_utils import UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS, PROJECTS, spms, datasets, BENCHMARK_DIR

BENCHMARK_ROOT = BENCHMARK_DIR.parent

NATIVE_CONFIG = {
    "af3": {
        "id_column": "sample",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["sample", "best_seed", "best_sample", "cognate_interaction",
                              "has_clash", "interface_hbonds"],
        "lower_is_better_columns": ["chain_pair_pae_min_0_1", "chain_pair_pae_min_1_0",
                                     "fraction_disordered"],
        "dhd": ("dhd_metrics_af3_filtered.csv", "_vs_"),
        "cop": ("cop_metrics_af3.csv", "_vs_"),
    },
    "boltz": {
        "id_column": "job_name",
        "cognate_column": "cognate_interaction",
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"],
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
        "unwanted_columns": ["job_name", "cognate_interaction", "interface_hbonds"],
        "lower_is_better_columns": ["intra_chain_1_pae", "intra_chain_0_pae", "inter_chain_pae"],
        "dhd": ("dhd_combined_metrics_filtered.csv", "__"),
        "cop": ("cop_combined_metrics.csv", "_vs_"),
    },
}


def combo_win_rate(oriented, combos):
    """Same worst-cognate-vs-best-non-cognate win check as pair_stats.py, but for a
    single combined score (e.g. the LDA projection) instead of one metric at a time."""
    successes = []
    for i, j, cross_rows in combos:
        values = [oriented.loc[i], oriented.loc[j]]
        for idx in cross_rows:
            values.append(oriented.loc[idx])

        has_missing = False
        for v in values:
            if pd.isna(v):
                has_missing = True
        if has_missing:
            continue

        worst_cognate = min(oriented.loc[i], oriented.loc[j])
        best_noncognate = oriented.loc[cross_rows[0]]
        for idx in cross_rows[1:]:
            val = oriented.loc[idx]
            if val > best_noncognate:
                best_noncognate = val
        successes.append(worst_cognate > best_noncognate)

    if not successes:
        return None
    return sum(successes) / len(successes)


def load_combined_metrics(spm, dataset):
    """Merge spm's native confidence metrics with fast_relax_riam's physics metrics,
    matched by which two proteins are paired. Returns one DataFrame with native_*
    columns, physics_* columns, plus protein_pair and cognate_status."""
    config = NATIVE_CONFIG[spm]
    filename, native_sep = config[dataset]
    native_df = pd.read_csv(BENCHMARK_ROOT / spm / "metrics" / filename)
    native_metric_columns = [c for c in native_df.columns if c not in config["unwanted_columns"]]

    native_data = native_df[native_metric_columns].copy()
    for col in config["lower_is_better_columns"]:
        if col in native_data.columns:
            native_data[col] = -native_data[col]
    native_data.columns = ["native_" + c for c in native_data.columns]
    native_data["_key"] = native_df[config["id_column"]].apply(lambda s: frozenset(s.split(native_sep)))
    native_data["cognate_label_native"] = native_df[config["cognate_column"]]

    physics_cfg = None
    for cfg in PROJECTS:
        if cfg.project == "fast_relax_riam" and cfg.spm == spm and cfg.dataset == dataset:
            physics_cfg = cfg
    physics_df = pd.read_csv(physics_cfg.metrics_path)
    physics_metric_columns = [c for c in physics_df.columns if c not in UNWANTED_COLUMNS]

    physics_data = physics_df[physics_metric_columns].copy()
    for col in LOWER_IS_BETTER_COLUMNS:
        if col in physics_data.columns:
            physics_data[col] = -physics_data[col]
    physics_data.columns = ["physics_" + c for c in physics_data.columns]
    physics_data["_key"] = physics_df["protein_pair"].apply(lambda s: frozenset(s.split("_vs_")))
    physics_data["protein_pair"] = physics_df["protein_pair"]
    physics_data["cognate_status"] = physics_df["cognate_status"]

    merged = native_data.merge(physics_data, on="_key", how="inner")

    n_mismatch = (merged["cognate_label_native"] != merged["cognate_status"]).sum()
    if n_mismatch:
        print(f"WARNING: {spm} {dataset}: {n_mismatch} rows disagree on cognate label "
              f"between native and physics sources")

    return merged.drop(columns=["_key", "cognate_label_native"])


for spm in spms:
    for dataset in datasets:
        label = f"combined_{spm}_{dataset}"
        results_dir = BENCHMARK_DIR / "results" / "lda_combined" / spm / dataset
        results_dir.mkdir(parents=True, exist_ok=True)

        merged = load_combined_metrics(spm, dataset)
        feature_columns = [c for c in merged.columns if c.startswith("native_") or c.startswith("physics_")]
        data = merged[feature_columns].copy()

        columns_with_nan = data.columns[data.isna().any()].tolist()
        if columns_with_nan:
            print(f"{label}: dropping metrics with missing data from LDA: {columns_with_nan}")
            data = data.drop(columns=columns_with_nan)

        data.index = merged["protein_pair"]

        X = StandardScaler().fit_transform(data.values)
        y = merged["cognate_status"].values

        # --- build combos for the win-rate check (same combo-finding logic as pair_stats.py) ---
        cognate_indices = merged.index[merged["cognate_status"] == 1].tolist()
        chain_sets = {}
        for i in cognate_indices:
            chain_sets[i] = set(merged.loc[i, "protein_pair"].split("_vs_"))

        combos = []
        for i, j in itertools.combinations(cognate_indices, 2):
            chains_i = chain_sets[i]
            chains_j = chain_sets[j]
            cross_rows = []
            for idx in merged.index:
                if idx == i or idx == j:
                    continue
                row_chains = set(merged.loc[idx, "protein_pair"].split("_vs_"))
                if len(row_chains & chains_i) == 1 and len(row_chains & chains_j) == 1:
                    cross_rows.append(idx)
            if cross_rows:
                combos.append((i, j, cross_rows))

        # --- fit LDA and score in-sample (fit and score on the same data -- optimistic) ---
        lda = LinearDiscriminantAnalysis()
        lda_scores = lda.fit_transform(X, y).ravel()
        in_sample_auc = roc_auc_score(y, lda_scores)
        print(f"--- {label} ---")
        print(f"n={len(y)}, cognate={int(y.sum())}, non-cognate={int(len(y) - y.sum())}, "
              f"n_metrics={len(data.columns)} ({sum(1 for c in data.columns if c.startswith('native_'))} native + "
              f"{sum(1 for c in data.columns if c.startswith('physics_'))} physics)")
        print(f"In-sample LDA AUC: {in_sample_auc:.3f}")

        cv = LeaveOneOut()
        cv_scores = cross_val_predict(lda, X, y, cv=cv, method='decision_function')
        cv_auc = roc_auc_score(y, cv_scores)
        print(f"Cross-validated LDA AUC: {cv_auc:.3f}")

        lda_score_series = pd.Series(lda_scores, index=merged.index)
        cv_score_series = pd.Series(cv_scores, index=merged.index)
        in_sample_win_rate = combo_win_rate(lda_score_series, combos)
        cv_win_rate = combo_win_rate(cv_score_series, combos)
        print(f"In-sample LDA win rate: {in_sample_win_rate:.0%} ({len(combos)} combos)")
        print(f"Cross-validated LDA win rate: {cv_win_rate:.0%} ({len(combos)} combos)")

        # --- save numeric result ---
        scores_df = pd.DataFrame({'protein_pair': data.index, 'lda_score': lda_scores, 'cv_lda_score': cv_scores})
        scores_df.to_csv(results_dir / f"{label}_lda_scores.csv", index=False)

        # --- 1. in-sample vs. cross-validated AUC and win rate ---
        bar_labels = ['In-sample\nAUC', 'Cross-validated\nAUC', 'In-sample\nwin rate', 'Cross-validated\nwin rate']
        bar_values = [in_sample_auc, cv_auc, in_sample_win_rate, cv_win_rate]
        bar_colors = ['steelblue', 'firebrick', 'steelblue', 'firebrick']

        fig, ax = plt.subplots(figsize=(7, 5.5))
        bars = ax.bar(bar_labels, bar_values, color=bar_colors)
        ax.axhline(0.5, color='gray', linestyle='--', label='Chance (0.5)')
        for bar, value in zip(bars, bar_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{value:.2f}",
                    ha='center', va='bottom')
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"LDA (native + physics): In-sample vs. Cross-validated — {label}")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
        plt.tight_layout()
        plt.savefig(results_dir / f"{label}_lda_auc_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # --- 2. ROC curve overlay: in-sample vs. cross-validated ---
        fpr_in, tpr_in, _ = roc_curve(y, lda_scores)
        fpr_cv, tpr_cv, _ = roc_curve(y, cv_scores)

        plt.figure(figsize=(6, 5))
        plt.plot(fpr_in, tpr_in, color='steelblue', label=f"In-sample (AUC={in_sample_auc:.2f})")
        plt.plot(fpr_cv, tpr_cv, color='firebrick', label=f"Cross-validated (AUC={cv_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"LDA (native + physics) ROC — {label}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / f"{label}_lda_roc_comparison.png", dpi=300)
        plt.close()
