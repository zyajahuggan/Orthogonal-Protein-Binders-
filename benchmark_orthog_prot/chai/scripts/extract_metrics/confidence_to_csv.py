import csv
import numpy as np
from pathlib import Path

output_path = Path("/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/chai_benchmark/outputs/cross_docking")
metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / f"{output_path.name}_metrics.csv"

metrics_path.parent.mkdir(parents=True, exist_ok=True)


def flatten_metrics(data):
    """Turn an npz file's arrays into flat scalar columns, e.g. per_chain_ptm -> per_chain_ptm_0, per_chain_ptm_1."""
    flat = {}
    for key in data.files:
        arr = np.squeeze(data[key])
        if arr.shape == ():
            flat[key] = arr.item()
        else:
            for idx in np.ndindex(arr.shape):
                suffix = "_".join(str(i) for i in idx)
                flat[f"{key}_{suffix}"] = arr[idx].item()
    return flat


rows = []
for project in sorted(output_path.iterdir()):
    score_file = project / "scores.model_idx_0.npz"
    if not score_file.exists():
        print(f"Skipping {project.name}: no scores file found")
        continue
    with np.load(score_file) as data:
        row = {"project": project.name}
        row.update(flatten_metrics(data))
    rows.append(row)

fieldnames = ["project"] + sorted({key for row in rows for key in row if key != "project"})

with open(metrics_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
