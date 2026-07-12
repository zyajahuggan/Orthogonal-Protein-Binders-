import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


metrics_path = Path(__file__).resolve().parent.parent.parent / "results" / "dhd" / "dhd_af3_interface_hbond_results.csv"
df = pd.read_csv(metrics_path)
cog_h_bonds = []
non_cog_h_bonds = []
for i in df.index:
    sample_name = df.loc[i,"filename"][:-10].split("_vs_")
    if sample_name[0][3:-1] in sample_name[1][3:-1] and sample_name[0][-1] != sample_name[1][-1]:
        cog_h_bonds.append(df.loc[i,"interface_hbonds"])
    else:
        non_cog_h_bonds.append(df.loc[i,"interface_hbonds"])

avg_h_bonds = {
    "cognant" : sum(cog_h_bonds)/len(cog_h_bonds), 
    "non_cognant" : sum(non_cog_h_bonds) / len(non_cog_h_bonds)
}
data = [cog_h_bonds, non_cog_h_bonds]
#making violin plot
fig, ax = plt.subplots()
vp = ax.violinplot(data,[2,4], widths=2,showmeans=True, showmedians=True, showextrema=True)

ax.set_xticks([2, 4])
ax.set_xticklabels(["Cognate", "non-cognate"])
ax.set_xlabel("Pair Type")           # overall x-axis label
ax.set_ylabel("Number of H-bonds")
ax.set_title("Interface H-bonds by Group")
plt.savefig(metrics_path.parent / "dhd_hbonds_violin.png", dpi=300, bbox_inches="tight")
plt.close(fig)



