import pandas as pd
from pathlib import Path

outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
spms = ["af3", "esmfold2", "chai", "boltz"]
datasets = ["dhd", "cross_docking"]
projects = ["riam", "fast_relax_riam"]
for project in projects:  
    for model in spms:
        for dataset in datasets:
            if project == "riam":
                per_task_csvs_dir = outputs_dir / project / model / dataset / "per_task"
            else:
                per_task_csvs_dir = outputs_dir / project / model / dataset / "per_task" / "per_task_csvs"
            csvs = list(per_task_csvs_dir.glob("*.csv"))
            print(len(csvs))
            if not csvs:
                print(f"No files found for {project}, skipping")
                continue

            df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
            out_path = outputs_dir.parent / "metrics" / f"{project}_{model}_{dataset}_combined.csv"
            df.to_csv(out_path, index=False)
            print(f"{project}: merged {len(csvs)} files -> {out_path} ({len(df)} rows)")