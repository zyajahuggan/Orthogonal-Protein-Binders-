from pathlib import Path
import pandas as pd
projects = "riam"
spms = ["af3", "esmfold2", "boltz", "chai"]
datasets = ["dhd", "cross_docking"]

for model in spms:
    for dataset in datasets:
        metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" / f"{projects}_{model}_{dataset}_combined.csv"
        df = pd.read_csv(metrics_path)

        for i in df.index:
            if model == "af3":
                df.loc[i, "protein_pair"] = df.loc[i, "protein_pair"][:-6]
            elif model == "esmfold2":
                df.loc[i, "protein_pair"] = df.loc[i, "protein_pair"].replace("__", "_vs_")
            elif model == "boltz":
                df.loc[i, "protein_pair"] = df.loc[i, "protein_pair"][:-8]
            # chai: no renaming needed

        df.to_csv(metrics_path, index=False)
        print(f"Updated {metrics_path}")


