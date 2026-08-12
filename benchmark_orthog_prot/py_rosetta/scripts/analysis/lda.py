"""Linear discriminant analysis: unlike PCA (unsupervised), LDA finds the single
direction through metric-space that best separates cognate from non-cognate pairs.
Cross-validated with leave-one-out since cognate pairs are few -- the gap between the
in-sample fit and the cross-validated score is the headline result: it shows whether
combining all metrics into one score actually generalizes, or just overfits a small
sample. Runs once per project (cop and dhd)."""

import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from pairing_utils import UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS, iter_datasets

ID_COL = "protein_pair"


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


for df, csv_stem, label, sep, results_dir in iter_datasets():
    lda_dir = results_dir / "lda"
    lda_dir.mkdir(parents=True, exist_ok=True)

    # --- select metric columns ---
    metric_columns = [c for c in df.columns if c not in UNWANTED_COLUMNS]

    # --- flip error-like metrics (lower is better -> higher is better) ---
    data = df[metric_columns].copy()
    for col in LOWER_IS_BETTER_COLUMNS:
        if col in data.columns:
            data[col] = -data[col]

    # LDA needs a complete matrix -- drop any metric column that has missing values
    # rather than crashing, since interface_loop_sc/interface_loop_sc_area are known
    # to have gaps in some projects (see topk_stats.py / score_margin.py)
    columns_with_nan = data.columns[data.isna().any()].tolist()
    if columns_with_nan:
        print(f"{label}: dropping metrics with missing data from LDA: {columns_with_nan}")
        data = data.drop(columns=columns_with_nan)

    data.index = df[ID_COL]  # pairs as rows, metrics as columns (correct LDA orientation)

    # --- standardize before LDA ---
    X = StandardScaler().fit_transform(data.values)
    y = df['cognate_status'].values

    # --- build combos for the win-rate check (same combo-finding logic as pair_stats.py) ---
    cognate_indices = df.index[df["cognate_status"] == 1].tolist()
    chain_sets = {}
    for i in cognate_indices:
        chain_sets[i] = set(df.loc[i, ID_COL].split(sep))

    combos = []
    for i, j in itertools.combinations(cognate_indices, 2):
        chains_i = chain_sets[i]
        chains_j = chain_sets[j]
        cross_rows = []
        for idx in df.index:
            if idx == i or idx == j:
                continue
            row_chains = set(df.loc[idx, ID_COL].split(sep))
            if len(row_chains & chains_i) == 1 and len(row_chains & chains_j) == 1:
                cross_rows.append(idx)
        if cross_rows:
            combos.append((i, j, cross_rows))

    # --- fit LDA and score in-sample (fit and score on the same data -- optimistic) ---
    lda = LinearDiscriminantAnalysis()
    lda_scores = lda.fit_transform(X, y).ravel()
    in_sample_auc = roc_auc_score(y, lda_scores)
    print(f"--- {label} ---")
    print(f"n={len(y)}, cognate={int(y.sum())}, non-cognate={int(len(y) - y.sum())}, n_metrics={len(metric_columns)}")
    print(f"In-sample LDA AUC: {in_sample_auc:.3f}")

    # given how few cognate pairs you likely have, LOO is probably the safer choice over k-fold
    cv = LeaveOneOut()
    cv_scores = cross_val_predict(lda, X, y, cv=cv, method='decision_function')
    cv_auc = roc_auc_score(y, cv_scores)
    print(f"Cross-validated LDA AUC: {cv_auc:.3f}")

    # win rate uses df's original row labels, so score the LDA output against that index
    # (not data.index, which was relabeled to the protein_pair id above)
    lda_score_series = pd.Series(lda_scores, index=df.index)
    cv_score_series = pd.Series(cv_scores, index=df.index)
    in_sample_win_rate = combo_win_rate(lda_score_series, combos)
    cv_win_rate = combo_win_rate(cv_score_series, combos)
    print(f"In-sample LDA win rate: {in_sample_win_rate:.0%} ({len(combos)} combos)")
    print(f"Cross-validated LDA win rate: {cv_win_rate:.0%} ({len(combos)} combos)")

    # --- save numeric result ---
    scores_df = pd.DataFrame({'protein_pair': data.index, 'lda_score': lda_scores, 'cv_lda_score': cv_scores})
    scores_df.to_csv(lda_dir / f"{csv_stem}_lda_scores.csv", index=False)

    # --- 1. in-sample vs. cross-validated AUC and win rate: the headline "looks good vs.
    # generalizes" comparison ---
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
    ax.set_title(f"LDA: In-sample vs. Cross-validated — {label}")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), frameon=False)
    plt.tight_layout()
    plt.savefig(lda_dir / f"{csv_stem}_lda_auc_comparison.png", dpi=300, bbox_inches='tight')
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
    plt.title(f"LDA ROC: In-sample vs. Cross-validated — {label}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(lda_dir / f"{csv_stem}_lda_roc_comparison.png", dpi=300)
    plt.close()
