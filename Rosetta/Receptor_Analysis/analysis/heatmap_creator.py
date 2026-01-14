import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Bio.PDB import PDBParser
import re


# ---------------------------------------------------------
# Extract WT amino acids from your WT PDB
# ---------------------------------------------------------
def extract_wt_from_pdb(pdb_file, chain_id="Y"):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("wt", pdb_file)

    three_to_one = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
        "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
        "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
        "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
    }

    wt_map = {}

    for chain in structure[0]:
        if chain.id != chain_id:
            continue

        for res in chain:
            if res.id[0] == " ":
                pos = res.id[1]
                aa = three_to_one.get(res.resname, "X")
                wt_map[int(pos)] = aa   # store with numeric pos as key

    return wt_map


# ---------------------------------------------------------
# Build matrices from CSV + WT PDB WT-labels
# ---------------------------------------------------------
def build_from_csv(csv_file, pdb_file, ddg_column="ddG_receptor"):

    # ---- Load CSV ----
    df = pd.read_csv(csv_file)

    # Parse mutation string: relaxed_wt_1_Y_132_A
    def parse(mut):
        parts = mut.split("_")
        chain = parts[-3]         # Y
        pos   = int(parts[-2])    # 132
        aa    = parts[-1]         # A
        return f"{chain}{pos}", aa, pos

    df["residue"], df["mutant"], df["pos"] = zip(*df["mutation"].apply(parse))

    # ---- Extract residues in numeric order ----
    residues = sorted(df["residue"].unique(), key=lambda r: int(r[1:]))

    # ---- All mutant amino acids present ----
    #amino_acids = sorted(df["mutant"].unique())
    AA_ORDER = [
    "G","A","V","L","I",
    "M","S","T","C","P",
    "N","Q","F","Y","W",
    "H","K","R","D","E"
    ]

    amino_acids = [aa for aa in AA_ORDER if aa in set(df["mutant"])]

    # ---- Extract true WT letters from PDB ----
    wt_map = extract_wt_from_pdb(pdb_file, chain_id="Y")

    # Build WT x-axis labels (e.g., E132)
    wt_labels = []
    for res in residues:
        pos = int(res[1:])
        wt_aa = wt_map.get(pos, "X")
        wt_labels.append(f"{wt_aa}{pos}")

    # ---- Build data matrix ----
    data = np.full((len(amino_acids), len(residues)), np.nan)
    annot = [["" for _ in residues] for _ in amino_acids]

    row_idx = {aa: i for i, aa in enumerate(amino_acids)}
    col_idx = {res: j for j, res in enumerate(residues)}

    # Fill numeric values
    for _, row in df.iterrows():
        r = row["residue"]
        aa = row["mutant"]
        val = row[ddg_column]

        data[row_idx[aa]][col_idx[r]] = val

    # ---- Annotate ONLY WT amino acid inside heatmap ----
    for j, res in enumerate(residues):
        pos = int(res[1:])
        wt_aa = wt_map.get(pos, "X")

        if wt_aa in row_idx:
            i = row_idx[wt_aa]
            annot[i][j] = wt_aa   # Only WT is labeled

    return data, annot, wt_labels, amino_acids


# ---------------------------------------------------------
# Plot heatmap
# ---------------------------------------------------------
def plot_heatmap(data, annot, wt_labels, amino_acids, outfile):

    fig, ax = plt.subplots(figsize=(50, 20))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "my_cmap",
        [(0, 0, 0.85), (1, 1, 1), (0.8, 0, 0)]
    )

    sns.set(font_scale=3.0)

    sns.heatmap(
        data,
        cmap=cmap,
        vmin=-10, vmax=10,
        linewidths=2,
        linecolor="black",
        annot=annot,            # Only WT labels appear
        fmt="s",
        annot_kws={"size":30, "fontweight": "bold"},
        xticklabels=wt_labels,  # WT labels from PDB
        yticklabels=amino_acids
    )

    #plt.xticks(rotation=45, ha="right",fontsize=20)
    ax = plt.gca()
    ax.set_xticks(np.arange(len(wt_labels)) + 0.9)   # center labels
    ax.set_xticklabels(wt_labels, rotation=45, ha="right", fontsize=20)
    plt.yticks(rotation=0,fontsize=40)
    plt.title("Receptor ΔΔG Heatmap", fontsize=50, pad=25)

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    csv_file = "/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/csv_files/relaxed_wt_1_receptor_avg.csv"
    pdb_file = "/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/structures/relaxed_wt_1.pdb"

    data, annot, labels, amino_acids = build_from_csv(csv_file, pdb_file)
    plot_heatmap(data, annot, labels, amino_acids, "receptor_heatmap.png")
