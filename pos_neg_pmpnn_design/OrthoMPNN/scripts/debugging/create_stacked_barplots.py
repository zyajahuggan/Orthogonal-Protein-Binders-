import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

path = Path("/home/zhuggan1/scr4_jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/OrthoMPNN/pdgf/output/csvs")

dfs = []

for f in path.glob("*.csv"):
    df = pd.read_csv(f)

    name = f.stem   # e.g., "CD_1.0_frequency"

    parts = name.split("_")

    group = parts[0]          # AB or CD
    concentration = parts[1]  # 1.0, 0.1, etc.

    df["group"] = group
    df["concentration"] = concentration

    # label used for facet title
    df["panel"] = f"{group} ({concentration})"

    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(data.head())

data_top = (
    data.sort_values("count", ascending=False)
        .groupby("panel")
        .head(20)           # <-- show top 20 per CSV
)

import seaborn as sns

# define the order explicitly
row_order = ["0.0", "0.1", "0.3", "0.5", "0.7", "1.0"]

g = sns.catplot(
    data=data,
    x="mutation",
    y="count",
    row="concentration",   # <-- rows by concentration
    col="group",           # <-- columns AB vs CD
    kind="bar",
    height=4.5,
    aspect=1.4,
    sharex=False,          # each subplot shows only its own mutations
    sharey=True,           # comparable counts across everything
    row_order=row_order    # enforce ordering
)

g.set_titles("{row_name} — {col_name}")
g.set_axis_labels("Mutation", "Counts")

for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelrotation=90)

g.tight_layout()
g.figure.savefig("faceted_by_group_and_concentration.png", dpi=300, bbox_inches="tight")
