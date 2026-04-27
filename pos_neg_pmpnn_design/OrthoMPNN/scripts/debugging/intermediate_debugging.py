import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# -----------------------
# Load all chain CSVs
# -----------------------
def chain_index_map():
    csv_dir = Path(
        "/scratch4/jgray21/zhuggan1/projects/orthosystems/"
        "tryingggg/pos_neg_design/OrthoMPNN/scripts"
    )
    csv_files = list(csv_dir.glob("chain*_index_map.csv"))

    dfs = [pd.read_csv(f) for f in csv_files]
    all_chains = pd.concat(dfs, ignore_index=True)

    # -----------------------
    # Build range summary
    # -----------------------
    range_summary = (
        all_chains
        .groupby("chain", sort=False)
        .agg(
            local_start=("local_idx", "min"),
            local_end=("local_idx", "max"),
            global_start=("global_idx", "min"),
            global_end=("global_idx", "max"),
            pdb_start=("pdb_idx", "min"),
            pdb_end=("pdb_idx", "max"),
        )
        .reset_index()
    )

    range_summary["length"] = (
        range_summary["local_end"] - range_summary["local_start"] + 1
    )

    # -----------------------
    # Compact range columns (presentation-friendly)
    # -----------------------
    range_summary["local_range"]  = (
        range_summary["local_start"].astype(str) + "–" + range_summary["local_end"].astype(str)
    )
    range_summary["global_range"] = (
        range_summary["global_start"].astype(str) + "–" + range_summary["global_end"].astype(str)
    )
    range_summary["pdb_range"]    = (
        range_summary["pdb_start"].astype(str) + "–" + range_summary["pdb_end"].astype(str)
    )

    range_summary = range_summary[
        ["chain", "local_range", "global_range", "pdb_range", "length"]
    ]

    # -----------------------
    # Sort chains in biological order
    # -----------------------
    chain_order = ["A", "B", "C", "D", "E", "F", "K", "L"]
    range_summary["chain"] = pd.Categorical(
        range_summary["chain"],
        categories=chain_order,
        ordered=True,
    )
    range_summary = range_summary.sort_values("chain")

    # -----------------------
    # Define chain groups (Option 3 coloring)
    # -----------------------
    ligand_chains   = {"A", "B", "E", "F"}
    receptor_chains = {"C", "D", "K", "L"}

    # -----------------------
    # Render table with matplotlib
    # -----------------------
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=range_summary.values,
        colLabels=range_summary.columns,
        loc="center",
        cellLoc="center",
    )

    # Font & spacing
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # -----------------------
    # Header styling
    # -----------------------
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    # -----------------------
    # Row coloring by chain group
    # -----------------------
    for i, chain in enumerate(range_summary["chain"], start=1):
        if chain in ligand_chains:
            color = "#e6f2ff"   # light blue
        elif chain in receptor_chains:
            color = "#e8f5e9"   # light green
        else:
            color = "white"

        for j in range(len(range_summary.columns)):
            table[(i, j)].set_facecolor(color)

    # -----------------------
    # Title & save
    # -----------------------
    plt.title("Chain Index Map", pad=20)

    plt.savefig(
        "chain_index_range_summary.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def interface_table():
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ------------------------------------------------------------
    # Load interface CSVs
    # ------------------------------------------------------------
    csv_dir = Path(
        "/scratch4/jgray21/zhuggan1/projects/orthosystems/"
        "tryingggg/pos_neg_design/OrthoMPNN/scripts"
    )

    csv_files = list(csv_dir.glob("chain*_pdb_interface.csv"))
    assert csv_files, "No chain*_pdb_interface.csv files found"

    dfs = [pd.read_csv(f) for f in csv_files]

    ## Force fresh copy
    all_df = pd.concat(dfs, ignore_index=True)

    # Drop old column if present
    if "orig_pdb_idx" in all_df.columns:
        all_df = all_df.drop(columns=["orig_pdb_idx"])

    orig_pdb_offset = {
        "A": 5,
        "B": -92,   # changed
        "C": -155,  # changed
    }

    all_df = all_df[all_df["chain"].isin(orig_pdb_offset)]

    all_df["orig_pdb_idx"] = (
        all_df["pdb_idx"].astype(int) +
        all_df["chain"].map(orig_pdb_offset).astype(int)
    )

    # Hard assertion
    for ch, off in orig_pdb_offset.items():
        delta = (
            all_df.loc[all_df["chain"] == ch, "orig_pdb_idx"]
            - all_df.loc[all_df["chain"] == ch, "pdb_idx"]
        ).unique()
        print(ch, delta)


    # ------------------------------------------------------------
    # Build interface table
    # ------------------------------------------------------------
    table_df = (
        all_df[all_df["is_interface"]]
        .loc[:, [
            "chain",
            "wt_aa",          # 👈 include WT amino acid
            "local_idx",
            "global_idx",
            "pdb_idx",
            "orig_pdb_idx",
        ]]
        .rename(columns={
            "pdb_idx": "design_pdb",
            "orig_pdb_idx": "original_pdb",
        })
        .sort_values(["chain", "design_pdb"])
    )

    # (Optional) limit rows if table is very long
    # table_df = table_df.head(30)

    # ------------------------------------------------------------
    # Render table as PNG
    # ------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(10, 0.4 * len(table_df) + 1)
    )
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.3)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f2f2f2")

    plt.title(
        "Interface residue mapping: design PDB → original PDB",
        pad=20,
    )

    plt.savefig(
        "interface_residue_mapping_table2.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

def tied_positions_positions():
    pdb_starts = {
        "A": 1,
        "B": 98,
        "C": 195,
        "D": 468,
        "E": 741,
        "F": 838,
        "K": 1675,
        "L": 1948,
    }

    import pandas as pd

    df = pd.read_csv("tied_positions_table.csv")

    chains_all = ["A","B","C","D","E","F","K","L"]

    def local_to_pdb(chain, local_idx):
        return pdb_starts[chain] + int(local_idx) - 1

    rows = []

    for tied_group, sub in df.groupby("tied_group"):
        # chains that actually participate in this tied group
        chains = [c for c in chains_all if sub[c].notna().any()]

        sub = sub.sort_values("tied_local")

        for _, r in sub.iterrows():
            out = {
                "tied_group": tied_group,
                "local": int(r["tied_local"]),
            }

            for ch in chains:
                local_val = int(r[ch])
                out[f"{ch}_local"] = local_val
                out[f"{ch}_pdb"] = local_to_pdb(ch, local_val)

            rows.append(out)

    proof_df = pd.DataFrame(rows)

    # --- FORCE INTEGER TYPES, BUT ALLOW NaNs ---
    for col in proof_df.columns:
        if col != "tied_group":
            proof_df[col] = proof_df[col].astype("Int64")

    print(proof_df.head())
    print(proof_df.tail())

    proof_df.to_csv("tied_local_pdb_full.csv", index=False)


    import matplotlib.pyplot as plt

    fig_height = 0.25 * len(proof_df) + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")

    tbl = ax.table(
        cellText=proof_df.values,
        colLabels=proof_df.columns,
        cellLoc="center",
        loc="center"
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.1)

    # header styling
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")

    plt.title("Tied-position Table", pad=12)
    plt.savefig("tied_local_pdb_full.png", dpi=300, bbox_inches="tight")
    plt.close()

def tiedgroupwithweights():
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---------------------------
    # Load data
    # ---------------------------
    df = pd.read_csv("tied_groups_with_weights.csv")

    # Enforce integer display (nullable-safe)
    for col in ["local_idx", "pdb_idx"]:
        df[col] = df[col].astype("Int64")

    # ---------------------------
    # Group positives and negatives
    # ---------------------------
    df_pos = df[df["weight"] > 0].sort_values(
        by=["tied_group", "local_idx", "chain"]
    )
    df_neg = df[df["weight"] < 0].sort_values(
        by=["tied_group", "local_idx", "chain"]
    )

    df_plot = pd.concat([df_pos, df_neg], ignore_index=True)

    n_pos = len(df_pos)

    # ---------------------------
    # Figure sizing
    # ---------------------------
    nrows = len(df_plot)
    fig_height = 0.28 * nrows + 2
    fig_width = 10

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    # ---------------------------
    # Build table
    # ---------------------------
    tbl = ax.table(
        cellText=df_plot.values,
        colLabels=df_plot.columns,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.1)

    # ---------------------------
    # Styling
    # ---------------------------
    for (r, c), cell in tbl.get_celld().items():

        # Header
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#E6E6E6")
            continue

        # Separator line between positive and negative blocks
        if r == n_pos:
            cell.set_edgecolor("black")
            cell.set_linewidth(1.5)

        colname = df_plot.columns[c]

        # Color weight column
        if colname == "weight":
            val = df_plot.iloc[r-1, c]
            cell.set_text_props(weight="bold")

            if val > 0:
                cell.set_facecolor("#D6E6FF")   # blue = positive design
            elif val < 0:
                cell.set_facecolor("#FFD6D6")   # red = negative design
            else:
                cell.set_facecolor("#F2F2F2")

    # ---------------------------
    # Title + legend
    # ---------------------------
    plt.title(
        "ProteinMPNN Tied Positions: Local / PDB Indexing with Design Weights",
        pad=14,
    )

    plt.text(
        0.01, -0.02,
        "Blue = positive design (+1.0)   |   Red = negative design (−neg_weight)",
        transform=ax.transAxes,
        fontsize=8,
    )

    # ---------------------------
    # Save PNG
    # ---------------------------
    plt.savefig(
        "tied_groups_with_weights_grouped.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

import pandas as pd
import numpy as np

df = pd.read_csv("fixedseedinterface_logits_rep0.csv")

# choose tied group
pos_chains = ["C", "D"]
neg_chains = ["K", "L"]

df_cd = df[df.chain.isin(pos_chains)]
df_kl = df[df.chain.isin(neg_chains)]

keys = ["local_idx", "aa"]


cd = (
    df_cd
    .groupby(keys)["delta_logit"]
    .mean()        # average C and D
    .rename("CD")
)

kl = (
    df_kl
    .groupby(keys)["delta_logit"]
    .mean()        # average K and L
    .rename("KL")
)

merged = pd.concat([cd, kl], axis=1).dropna()

merged["CD_minus_KL"] = merged["CD"] - merged["KL"]

import matplotlib.pyplot as plt

heatmap = merged["CD_minus_KL"].unstack("local_idx")


# optional AA order
aa_order = list("ACDEFGHIKLMNPQRSTVWY")
heatmap = heatmap.loc[[aa for aa in aa_order if aa in heatmap.index]]

fig, ax = plt.subplots(
    figsize=(0.35 * heatmap.shape[1] + 3, 6)
)

im = ax.imshow(
    heatmap.values,
    aspect="auto",
    cmap="coolwarm",
    vmin=-np.nanmax(np.abs(heatmap.values)),
    vmax= np.nanmax(np.abs(heatmap.values)),
)

ax.set_yticks(range(len(heatmap.index)))
ax.set_yticklabels(heatmap.index)

ax.set_xticks(range(len(heatmap.columns)))
ax.set_xticklabels(heatmap.columns, rotation=90)

ax.set_xlabel("Interface residues (PDB index)")
ax.set_ylabel("Amino acids")
ax.set_title("Orthogonality heatmap: (C,D) − (K,L)")

plt.colorbar(im, ax=ax, label="ΔΔlogit (CD − KL)")
plt.tight_layout()
plt.savefig("fixeddecodingCD_vs_KL_delta_delta_logit_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


