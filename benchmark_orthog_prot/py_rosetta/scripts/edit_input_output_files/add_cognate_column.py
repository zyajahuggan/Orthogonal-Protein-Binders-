import pandas as pd
from pathlib import Path
import re 

project = "riam"
spms = ["af3", "esmfold2", "chai", "boltz"]
dataset = "dhd"



for model in spms:
    metrics_path = Path(__file__).resolve().parent.parent.parent / "metrics" /f"{project}_{model}_{dataset}_combined.csv"
    df = pd.read_csv(metrics_path)
    cognate_status = []
    pattern = re.compile(r"^(.+)(?:a_vs_\1b|b_vs_\1a)$")
    for job in df['protein_pair']:
        found = pattern.search(job)
        if found is None:
            cognate_status.append(0)
        else:
            cognate_status.append(1)
    print(cognate_status)
    print(sum(cognate_status))
    df["cognate_interaction"] = cognate_status
    df.to_csv(metrics_path, index = False)

# def get_cognate_status_dhd(name):
#     cognate_status = 0
#     pattern = re.compile(r"^(.+)(?:a_vs_\1b|b_vs_\1a)$")
#     found = pattern.search(name)
#     if found is not None:
#         cognate_status = 1
#     return cognate_status


