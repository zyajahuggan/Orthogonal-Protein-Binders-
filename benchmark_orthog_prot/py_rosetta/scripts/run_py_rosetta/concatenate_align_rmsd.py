import pandas as pd
from pathlib import Path

project = "fast_relax_riam"
spms = ["af3", "boltz", "esmfold2", "chai"]
datasets = ["dhd", "cross_docking"]

base_dir = Path(__file__).resolve().parent.parent.parent

for dataset in datasets:
    csvs = []
    for model in spms:
        per_task_csv_dir = base_dir / "outputs" / project / model / dataset / "per_task" / "align_rmsd_csvs"
        csvs.extend(per_task_csv_dir.glob("*.csv"))

    if not csvs:
        print(f"No align_rmsd csvs found for {dataset}, skipping")
        continue

    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    out_path = base_dir / "metrics" / f"{project}_{dataset}_align_rmsd.csv"
    df.to_csv(out_path, index=False)
    print(f"{dataset}: merged {len(csvs)} files -> {out_path} ({len(df)} rows)")
