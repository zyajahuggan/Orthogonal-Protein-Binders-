import os 
import sys 
import pandas as pd 
import matplotlib.pyplot as plt 



orig_pdb_offset = {
    "A": 5,
    "C": -155,
}

import re

def apply_pdb_offset(mutation, offset_dict):
    """
    Example input:  'A:R74Q'
    Example output: 'A:R79Q'  (if offset[A] = +5)
    """
    try:
        chain, rest = mutation.split(":")
        m = re.match(r"([A-Z])(\d+)([A-Z])", rest)
        if not m:
            return mutation  # fail safely

        wt, resi, mut = m.groups()
        resi = int(resi)

        offset = offset_dict.get(chain, 0)
        new_resi = resi + offset

        return f"{chain}:{wt}{new_resi}{mut}"

    except Exception:
        return mutation


directory_path = '/scratch4/jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/OrthoMPNN/scripts'

for name in os.listdir(directory_path):
    if name.startswith(("AB_", "CD_")) and name.endswith(".csv"):
        full = os.path.join(directory_path,name)
        df = pd.read_csv(full)

        df["mutation"] = df["mutation"].apply(
            lambda m: apply_pdb_offset(m, orig_pdb_offset)
        )
        df = df.sort_values(by="best_delta", key=lambda x:x.abs())

        aa_property = {
            'G': 'nonpolar',
            'A': 'nonpolar',
            'V': 'nonpolar',
            'I': 'nonpolar',
            'L': 'nonpolar',
            'M': 'nonpolar',

            'P': 'polar',
            'T': 'polar',
            'S': 'polar',
            'N': 'polar',
            'Q': 'polar',
            'C': 'polar',

            'K': 'positive',
            'R': 'positive',
            'H': 'positive',

            'D': 'negative',
            'E': 'negative',

            'W': 'aromatic',
            'Y': 'aromatic',
            'F': 'aromatic',
        }

        df['mut_aa'] = df['mutation'].str[-1]
        df['aa_property'] = df['mut_aa'].map(aa_property)

        df = df.dropna(subset=["aa_property"])
        df['aa_property'] = pd.Categorical(
            df['aa_property'],
            categories=['nonpolar','polar','positive','negative','aromatic'],
            ordered=True,

        )

        prop_to_color = {
            "nonpolar": "C0",  # blue
            "polar":      "C1",   # orange
            "positive":   "C2",   # green
            "negative":   "C3",   # red
            "aromatic":   "C4",   # purple
        }

        colors = df["aa_property"].map(prop_to_color).to_numpy()

        plt.figure()

        plt.scatter(
            df["mutation"],
            df["best_delta"],
            c=colors,
            alpha=0.85,
        )
        plt.ylabel('Δ logP')
        plt.xlabel('Mutation')

        weight = float(name.split("stats")[-1].replace(".csv", ""))
        
        ax = plt.gca()

        if weight == 1.0:
            # show every other label
            ax.set_xticks(ax.get_xticks()[::14])
            ax.tick_params(axis="x", labelsize=6)
        elif weight == 0.6:
            ax.set_xticks(ax.get_xticks()[::2])
            ax.tick_params(axis="x", labelsize=5)
        else:
            ax.tick_params(axis="x", labelsize=5)

        plt.xticks(rotation=90)

        plt.axhline(0, linestyle='--',linewidth=1)

        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="", color=prop_to_color[p], label=p)
            for p in df["aa_property"].cat.categories
        ]
        plt.legend(handles=handles, title="AA property", loc="upper right")

        system = name.split("_")[0]

        plt.title(f"{system} Mutation Differences (weight = {weight})")
        plt.tight_layout()
        plt.savefig(f'{full}.png')
        plt.show()
        plt.close()