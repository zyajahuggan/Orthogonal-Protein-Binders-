import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / "dhd_metrics_af3_filtered.csv"
df = pd.read_csv(metrics_path)

csv_stem = metrics_path.stem
project = "dhd" if "dhd" in csv_stem else "cross_docking"
results_dir = metrics_path.parent.parent / "results" / project
results_dir.mkdir(parents=True, exist_ok=True)

pae_columns = ['chain_pair_pae_min_0_1', 'chain_pair_pae_min_1_0']
unwanted_columns = ["sample", "best_seed", "best_sample", "cognate_interaction", "has_clash"]

metric_columns = [c for c in df.columns if c not in unwanted_columns]

data = df[metric_columns].copy()
for col in pae_columns:
    if col in data.columns:
        data[col] = -data[col]

data.index = df['sample']

# --- standardize columns (metrics), since pairs are now rows ---
X = StandardScaler().fit_transform(data.values)   # rows = pairs, columns = metrics

# --- covariance matrix across metrics, using pairs as observations ---
cov_matrix = np.cov(X, rowvar=False)   # rowvar=False tells numpy that columns are variables

# --- eigen-decomposition ---
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# sort descending
sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]

explained_variance_ratio = eigenvalues / eigenvalues.sum()

# --- project pairs onto the top components ---
pc_scores = X @ eigenvectors[:, :2]
df['PC1'] = pc_scores[:, 0]
df['PC2'] = pc_scores[:, 1]

#silouhette score to determine k, number of clusters
# for k in range(2, 8):
#     km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(pc_scores)
#     score = silhouette_score(pc_scores, km.labels_)
#     print(f"k={k}: silhouette={score:.3f}")



# use however many components you want to retain (e.g., first 2-3, since PC1+PC2 = 87% variance)
n_components = 2
pc_scores = X @ eigenvectors[:, :n_components]

# --- k-means clustering in PC space ---
k = 2 
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(pc_scores)

df['cluster'] = cluster_labels

# --- visualize clusters in PC1/PC2 space, colored by cluster and shaped by cognate status ---
plt.figure(figsize=(7, 6))
for c in range(k):
    mask = df['cluster'] == c
    plt.scatter(pc_scores[mask, 0], pc_scores[mask, 1], label=f'Cluster {c}', alpha=0.7)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.title("Protein Pairs Clustered in PCA Space")
plt.savefig(results_dir / "pca_clusters.png", dpi=300)
plt.close()

# print(df.groupby('cluster')['cognate_interaction'].value_counts(normalize=True))
# print(df.groupby('cluster')['cognate_interaction'].value_counts())  # raw counts too

# what actually distinguishes the two clusters?
# print(df.groupby('cluster')[metric_columns].mean())
# print(df['cognate_interaction'].value_counts(normalize=True))

import numpy as np

centroids = kmeans.cluster_centers_
df['dist_to_centroid'] = [
    np.linalg.norm(pc_scores[i] - centroids[df['cluster'].iloc[i]])
    for i in range(len(df))
]

# most "typical" examples per cluster
print(df.sort_values('dist_to_centroid').groupby('cluster').head(3)[['sample', 'cluster', 'cognate_interaction']])

# the interesting misclassified cases
print(df[(df['cluster'] == 0) & (df['cognate_interaction'] == 1)][['sample', 'cluster']])
print(df[(df['cluster'] == 1) & (df['cognate_interaction'] == 0)][['sample', 'cluster']])