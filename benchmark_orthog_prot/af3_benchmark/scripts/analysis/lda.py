import pandas as pd
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- paths / setup ---
metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "dhd_metrics_af3_filtered.csv"
df = pd.read_csv(metrics_path)

csv_stem = metrics_path.stem
project = "dhd" if "dhd" in csv_stem else "cross_docking"
results_dir = metrics_path.parent.parent / "results" / project
results_dir.mkdir(parents=True, exist_ok=True)

pae_columns = ['chain_pair_pae_min_0_1', 'chain_pair_pae_min_1_0']
unwanted_columns = ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash"]

# --- select metric columns ---
metric_columns = [c for c in df.columns if c not in unwanted_columns]

# --- flip PAE columns (lower is better -> higher is better) ---
data = df[metric_columns].copy()
for col in pae_columns:
    if col in data.columns:
        data[col] = -data[col]

data.index = df['sample']  # pairs as rows, metrics as columns (correct LDA orientation)

# --- standardize before LDA ---
X = StandardScaler().fit_transform(data.values)
y = df['cognate_interaction'].values

# --- fit LDA and score ---
lda = LinearDiscriminantAnalysis()
lda_scores = lda.fit_transform(X, y).ravel()

auc = roc_auc_score(y, lda_scores)
print(f"LDA AUC: {auc:.3f}")

from sklearn.model_selection import LeaveOneOut, cross_val_predict

# given how few cognate pairs you likely have, LOO is probably the safer choice over k-fold
cv = LeaveOneOut()
cv_scores = cross_val_predict(lda, X, y, cv=cv, method='decision_function')

cv_auc = roc_auc_score(y, cv_scores)
print(f"Cross-validated LDA AUC: {cv_auc:.3f}")
# --- save result ---
pd.DataFrame({'sample': data.index, 'lda_score': lda_scores}).to_csv()