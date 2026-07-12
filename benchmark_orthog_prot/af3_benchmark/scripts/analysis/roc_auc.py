import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from pathlib import Path
import matplotlib.pyplot as plt
from pairing_utils import orient_scores, UNWANTED_COLUMNS, LOWER_IS_BETTER_COLUMNS

metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "cross_docking_metrics_af3.csv"
df = pd.read_csv(metrics_path)

csv_stem = metrics_path.stem
project = "dhd" if "dhd" in csv_stem else "cross_docking"
results_dir = metrics_path.parent.parent / "results" / project

# --- AUC leaderboard
results = []
for metric in df.columns:
    if metric.strip() in UNWANTED_COLUMNS:
        continue
    scores = orient_scores(df, metric)
    auc = roc_auc_score(df['cognate_interaction'], scores)
    results.append({'metric': metric, 'auc': auc})

leaderboard = pd.DataFrame(results).sort_values('auc', ascending=False)
leaderboard.to_csv(results_dir / f"af3_auc_{project}_leaderboard.csv", index=False)

# --- output dirs ---
roc_dir = results_dir / "roc_curves"
roc_dir.mkdir(parents=True, exist_ok=True)

youden_dir = results_dir / "youdens_j" / csv_stem
youden_dir.mkdir(parents=True, exist_ok=True)

barplot_dir = results_dir / "barplots"
barplot_dir.mkdir(parents=True, exist_ok=True)

# --- merged loop: ROC plot + Youden's J, once per metric ---
youden_results = []
for metric in leaderboard['metric']:
    scores = orient_scores(df, metric)
    fpr, tpr, thresholds = roc_curve(df['cognate_interaction'], scores)
    auc = roc_auc_score(df['cognate_interaction'], scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{metric} (AUC={auc:.2f})", color='steelblue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {metric}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_dir / f"roc_{metric}.png", dpi=300)
    plt.close()

    j_scores = tpr - fpr
    best_idx = j_scores.argmax()
    best_threshold = thresholds[best_idx]
    if metric in LOWER_IS_BETTER_COLUMNS:
        best_threshold = -best_threshold

    youden_results.append({
        'metric': metric,
        'best_threshold': best_threshold,
        'tpr_at_best': tpr[best_idx],
        'fpr_at_best': fpr[best_idx],
        'youden_j': j_scores[best_idx]
    })

youden_df = pd.DataFrame(youden_results).sort_values('youden_j', ascending=False)
youden_df.to_csv(youden_dir / f"youden_thresholds_{project}.csv", index=False)

# --- AUC bar chart ---
plt.figure(figsize=(8, 5))
plt.barh(leaderboard['metric'], leaderboard['auc'])
plt.xlabel("AUC")
plt.title("AUC by Metric — AF3")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(barplot_dir / f"af3_auc_bar_{project}.png", dpi=300)
plt.close()




