import pandas as pd
from pathlib import Path
project = "riam"
spms = ["af3", "esmfold2", "chai", "boltz"]
dataset = "cop"
for model in spms:
    metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / f"{project}_{model}_{dataset}_combined.csv"
    df = pd.read_csv(metrics_path)
    cognate_status = []
    for job in df["protein_pair"]:
        ligand, receptor = job.split("_vs_")  # tuple unpacking
        if ligand + "s" == receptor:
            cognate_status.append(1)
        else:
            cognate_status.append(0)
    print(sum(cognate_status))
    df["cognate_interaction"] = cognate_status
    df.to_csv(metrics_path, index=False)
    
def get_cognate_status_cop(design_name):
    ligand, receptor = design_name.split("_vs_")
    cognate_status = 0 
    if ligand + "s" == receptor:
            cognate_status = 1
    return cognate_status