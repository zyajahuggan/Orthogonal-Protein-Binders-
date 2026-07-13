"""Linear discriminant analysis: unlike PCA (unsupervised), LDA finds the single
direction through metric-space that best separates cognate from non-cognate pairs.
Cross-validated with leave-one-out since cognate pairs are few -- the gap between the
in-sample fit and the cross-validated score is the headline result: it shows whether
combining all metrics into one score actually generalizes, or just overfits a small
sample. Runs once per project (cross_docking and dhd)."""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from pairing_utils import UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS, iter_datasets

MODEL_NAME = "AF3"
ID_COL = "sample"

for df, csv_stem, project, sep, results_dir in iter_datasets():
    lda_dir = results_dir / "lda"
    lda_dir.mkdir(parents=True, exist_ok=True)

    # --- select metric columns ---
    metric_columns = [c for c in df.columns if c not in UNWANTED_COLUMNS]

    # --- flip error-like metrics (lower is better -> higher is better) ---
    data = df[metric_columns].copy()
    for col in LOWER_IS_BETTER_COLUMNS:
        if col in data.columns:
            data[col] = -data[col]

    data.index = df[ID_COL]  # pairs as rows, metrics as columns (correct LDA orientation)

    assert not data.isna().any().any(), \
        f"NaNs found in metric columns for {project}: {data.columns[data.isna().any()].tolist()}"

    # --- standardize before LDA ---
    X = StandardScaler().fit_transform(data.values)
    y = df['cognate_interaction'].values

    # --- fit LDA and score in-sample (fit and score on the same data -- optimistic) ---
    lda = LinearDiscriminantAnalysis()
    lda_scores = lda.fit_transform(X, y).ravel()
    in_sample_auc = roc_auc_score(y, lda_scores)
    print(f"--- {project} ---")
    print(f"n={len(y)}, cognate={int(y.sum())}, non-cognate={int(len(y) - y.sum())}, n_metrics={len(metric_columns)}")
    print(f"In-sample LDA AUC: {in_sample_auc:.3f}")

    # given how few cognate pairs you likely have, LOO is probably the safer choice over k-fold
    cv = LeaveOneOut()
    cv_scores = cross_val_predict(lda, X, y, cv=cv, method='decision_function')
    cv_auc = roc_auc_score(y, cv_scores)
    print(f"Cross-validated LDA AUC: {cv_auc:.3f}")

    # --- save numeric result ---
    scores_df = pd.DataFrame({'sample': data.index, 'lda_score': lda_scores, 'cv_lda_score': cv_scores})
    scores_df.to_csv(lda_dir / f"{csv_stem}_lda_scores.csv", index=False)

    # --- 1. in-sample vs. cross-validated AUC bar chart: the headline "looks good vs.
    # generalizes" comparison ---
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    bars = ax.bar(['In-sample', 'Cross-validated'], [in_sample_auc, cv_auc],
                   color=['steelblue', 'firebrick'])
    ax.axhline(0.5, color='gray', linestyle='--', label='Chance (AUC=0.5)')
    for bar, auc in zip(bars, [in_sample_auc, cv_auc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{auc:.3f}",
                ha='center', va='bottom')
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"LDA: In-sample vs. Cross-validated AUC\n{MODEL_NAME} ({project})")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.08), frameon=False)
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
    plt.title(f"LDA ROC: In-sample vs. Cross-validated — {MODEL_NAME} ({project})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(lda_dir / f"{csv_stem}_lda_roc_comparison.png", dpi=300)
    plt.close()
