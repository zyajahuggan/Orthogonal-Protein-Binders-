import numpy as np
from pathlib import Path

output_path = Path("/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/chai_benchmark/outputs/dhd")
metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / f"{output_path.name}_metrics.csv"

metrics_path.parent.mkdir(parents=True, exist_ok=True)

with open(metrics_path, 'w') as file:
    for project in output_path.iterdir():
        score_file = project / "scores.model_idx_0.npz"
        with np.load(score_file) as data:
            metrics = {}
            for i in range(len(data.files)):
                metrics[data.files[i]] = data[data.files[i]]
        file.write(f"{project.name}__{metrics}\n")


    
    




