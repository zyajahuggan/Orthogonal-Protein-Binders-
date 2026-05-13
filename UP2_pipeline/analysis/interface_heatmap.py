import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.cm import ScalarMappable
from Bio.PDB import PDBParser
import re
import os

def load_sc_file(scorefile):
    '''Load Rosetta .sc file (with replicate averaging)'''
    df = pd.read_csv(scorefile, sep=r"\s+", comment="#") # regex pattern for white space for columns, skip lines that have # 

    def parse_mutation(desc):
        m = re.search(r".*_(\w)_(\d+)_(\w)(?:_rep\d+)?\.pdb", desc) # breaks down name of pdb files,  \w one word char, \d one or more digits
        if m:
            chain, pos, mut = m.groups() # unpack regex tuple 
            return f"{chain}{pos}", mut # ex: (A45, K)
        else:
            return "WT", "WT" # identifies wild type structure 

    df[["residue", "mutant"]] = df["description"].apply(parse_mutation).apply(pd.Series) # "myprotein_A_45_K.pdb"  --> ("A45", "K") --> "A45"  |  "K"

    # --- Average over replicates ---
    energy_cols = [c for c in df.columns if c.startswith("dG") or c.startswith("ddG")] # keep only dG and ddG columns 
    df_avg = (
        df.groupby(["residue","mutant"], as_index=False)[energy_cols]
          .mean()
    ) # group all replicates together and average all of them together 
    return df_avg

def get_wt_map_and_labels(wt_pdb, residues_chainpos):
    '''
    Build a mapping of residue positions to their wild-type amino acids
    and a list of labels for the heatmap
    residues_chainpos: list of residue positions (e.g. ["A12", "A45", "A67", "B23"])
    '''
    parser = PDBParser(QUIET=True) # biopython parser
    structure = parser.get_structure("wt", wt_pdb) 

    one_letter = {
        "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F",
        "GLY":"G","HIS":"H","ILE":"I","LYS":"K","LEU":"L",
        "MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R",
        "SER":"S","THR":"T","VAL":"V","TRP":"W","TYR":"Y"
    }
    # Biopython Structure 
    # Structure
    # └── Model [0]
    #     └── Chain A 
    #         └── Residue 45
    #             └── Atom CA
    wt_map = {}
    for chain in structure[0]:
        for res in chain:
            if res.id[0] == " ": # checks for protein residues 
                pos = res.id[1]
                wt_aa = one_letter.get(res.resname, "X")
                wt_map[f"{chain.id}{pos}"] = wt_aa # wt_map["A45"] = "K"

    wt_labels = [] # stores heatmap labels 
    for key in residues_chainpos:
        if key in wt_map:
            pos = re.search(r"\d+", key).group() # get position number from key
            wt_labels.append(f"{wt_map[key]}{pos}") # combines wt amino acid with position number
        else:
            wt_labels.append(key)
    return wt_map, wt_labels

def build_ddg_array(df, residues_chainpos, amino_acid, col_idx, row_idx, col_name="ddG_interface"):
    '''
    Build array of ddG values for the heatmap
    residues_chainpos: list of residue positions (e.g. ["A12", "A45", "A67", "B23"])
    amino_acid: list of amino acids (e.g. ["G", "A", "V"])
    col_idx: mapping of column indices for residues
    row_idx: mapping of row indices for amino acids
    col_name: name of the column in df to use for ddG values
    '''
    arr = np.full((len(amino_acid), len(residues_chainpos)), np.nan) # np.full creates numpy array and fulls every element with same value(20 x # res in interaction sphere)
    for _, row in df.iterrows(): # iterrows retuns index and row data
        res_key, mut = row["residue"], row["mutant"]
        if res_key == "WT" or res_key not in col_idx or mut not in row_idx:
            continue
        j = col_idx[res_key]
        i = row_idx[mut]
        arr[i, j] = row[col_name] # place ddg value into correct position 

        #        A12    A45    A67    B23
        # G      NaN    2.3    NaN    NaN
        # A      NaN    NaN    NaN    -1.2
        # V      NaN    NaN    0.8    NaN
    return arr

# ---------- Plot one triangular heatmap ----------
def plot_triangle_on_ax(ax, bottom_array, top_array, residues_chainpos,
                        wt_labels, amino_acid,
                        norm_red, cmap_red,
                        norm_blue, cmap_blue, keep_cols):

   
    # arrays based on kept columns
    bottom_array = bottom_array[:, keep_cols]
    top_array = top_array[:, keep_cols]
    residues_chainpos = np.array(residues_chainpos)[keep_cols]
    wt_labels = np.array(wt_labels)[keep_cols]

    M = len(residues_chainpos) # number of residues after filtering 
    N = len(amino_acid)
    x = np.arange(M + 1) # columns 
    y = np.arange(N + 1) # rows 
    xs, ys = np.meshgrid(x, y) # 2D grid of every point 

    #(0,3)-(1,3)-(2,3)
    #|     |     |
    #(0,2)-(1,2)-(2,2)
    #|     |     |
    #(0,1)-(1,1)-(2,1)
    #|     |     |
    #(0,0)-(1,0)-(2,0)

    triangles_bottom = [
        (i + (j+1)*(M+1), i+1 + (j+1)*(M+1), i + j*(M+1))
        for j in range(N) for i in range(M)
    ]
    triangles_top = [
        (i+1 + j*(M+1), i+1 + (j+1)*(M+1), i + j*(M+1))
        for j in range(N) for i in range(M)
    ]

    # Flattened list of x and y coordinates
    triang_bottom = Triangulation(xs.ravel(), ys.ravel(), triangles_bottom)
    triang_top    = Triangulation(xs.ravel(), ys.ravel(), triangles_top)

    # List of ddg values 
    c_bottom = np.ma.masked_invalid(bottom_array).ravel(order="C")
    c_top    = np.ma.masked_invalid(top_array).ravel(order="C")

    # Bottom (WT) in Reds
    ax.tripcolor(triang_bottom, c_bottom, shading="flat",
                 cmap=cmap_red, edgecolors="white", linewidths=0.5,
                 norm=norm_red)

    # Top (Ortho) in Blues
    ax.tripcolor(triang_top, c_top, shading="flat",
                 cmap=cmap_blue, edgecolors="white", linewidths=0.5,
                 norm=norm_blue)

    # Arranges placement of labels of heatmap
    ax.set_xticks(np.arange(M) + 0.5)
    ax.set_yticks(np.arange(N) + 0.5)

    # Places Label on respective axis 
    ax.set_xticklabels(wt_labels, rotation=45, fontsize=12)
    ax.set_yticklabels(amino_acid, fontsize=12)

    ax.set_xlim(0, M)
    ax.set_ylim(0, N)
    ax.invert_yaxis()
    ax.set_aspect("equal") # square

    # Show mean values for each column above heatmap 
    # --- Show both A and E (stacked, color-coded, 0–5 range consistent with colorbars) ---
    vmax_red  = getattr(norm_red,  "vmax", 5.0)
    vmin_blue = getattr(norm_blue, "vmin", -5.0)

    # Ablation (red): keep positives, clip to [0, vmax_red]
    abl_clip = np.clip(bottom_array, 0, None)
    abl_clip = np.minimum(abl_clip, vmax_red)

    # Enrichment (blue): keep negatives, take magnitude, clip to [0, |vmin_blue|]
    enr_clip = np.clip(top_array, None, 0)
    enr_clip = -enr_clip
    enr_clip = np.minimum(enr_clip, abs(vmin_blue))

    mean_ablation   = np.nanmean(abl_clip, axis=0)
    mean_enrichment = np.nanmean(enr_clip, axis=0)

    # Use axis transform so labels stay just above the plot
    t = ax.get_xaxis_transform()
    yA = 1.025   # A (red) slightly above top
    yE = 1.010   # E (blue) just below A

    for i, (mA, mE) in enumerate(zip(mean_ablation, mean_enrichment)):
        # A label (red)
        ax.text(i + 0.5, yA, f"A:{mA:.1f}",
                transform=t, clip_on=False,
                ha="center", va="bottom",
                fontsize=18, rotation=0,
                fontweight="bold", color="darkred",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none",
                        alpha=0.8, pad=0.2))

        # E label (blue, directly beneath A)
        ax.text(i + 0.5, yE, f"E:{mE:.1f}",
                transform=t, clip_on=False,
                ha="center", va="bottom",
                fontsize=18, rotation=0,
                fontweight="bold", color="darkblue",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="none",
                        alpha=0.8, pad=0.2))






def plot_split_heatmaps(wt_A, ortho_A, residues_A, wt_labels_A,
                        wt_B, ortho_B, residues_B, wt_labels_B,
                        amino_acid,
                        energy_term="ddG_interface",
                        outfile="split_raw_interface.png",
                        label_red="WT ΔΔG", label_blue="Ortho ΔΔG", ablation_threshold=0):

    import matplotlib.gridspec as gridspec

    # --- Wider figure for better aspect ratio ---
    fig = plt.figure(figsize=(70, 25))

    # constructing actual heatmap 
    mean_abl_A = np.nanmean(np.clip(wt_A, 0, None), axis=0) # mean of ablation columns to determine which columns to keep 
    keep_A = mean_abl_A >= ablation_threshold

    mean_abl_B = np.nanmean(np.clip(wt_B, 0, None), axis=0)
    keep_B = mean_abl_B >= ablation_threshold

    n_A = int(np.sum(keep_A))
    n_B = int(np.sum(keep_B))

    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[n_A, 0.6, n_B],
        wspace=0.05
    )

    axA = fig.add_subplot(gs[0, 0])
    ax_divider = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[0, 2])
    axes = [axA, axB]

    # --- Colormaps ---
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    # WT: stabilizing = green, neutral = white, destabilizing = red
    cmap_red = LinearSegmentedColormap.from_list(
        "wt_map",
        ["blue", "white", "red"]
    )

    # custom colormap, ORTHO: favorable = blue, neutral = white, wrong-way = green
    cmap_blue = LinearSegmentedColormap.from_list(
        "ortho_map",
        ["blue", "white", "red"]
    )

    cmap_red.set_bad("lightgray")
    cmap_blue.set_bad("lightgray")

    norm_red = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    norm_blue = TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)

    # --- Draw Chain A ---
    plot_triangle_on_ax(axA, wt_A, ortho_A, residues_A, wt_labels_A,
                        amino_acid, norm_red, cmap_red, norm_blue, cmap_blue, keep_A)
    #axA.set_title(f"Chain A {energy_term}", fontsize=26)

    # --- Draw Chain B ---
    plot_triangle_on_ax(axB, wt_B, ortho_B, residues_B, wt_labels_B,
                        amino_acid, norm_red, cmap_red, norm_blue, cmap_blue, keep_B)
    #axB.set_title(f"Chain B {energy_term}", fontsize=26)

    # --- Black divider ---
    ax_divider.set_facecolor("black")
    ax_divider.set_xticks([]); ax_divider.set_yticks([])
    for spine in ax_divider.spines.values():
        spine.set_visible(False)

    # --- Match vertical heights ---
    ylim0, ylim1 = axA.get_ylim(), axB.get_ylim()
    min_y, max_y = min(ylim0[0], ylim1[0]), max(ylim0[1], ylim1[1])
    axA.set_ylim(min_y, max_y)
    axB.set_ylim(min_y, max_y)

    # --- Style axes ---
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=40, width=1.5)
        ax.title.set_fontsize(60)
        ax.set_facecolor("white")

    # --- Figure background ---
    fig.patch.set_facecolor("white")

    # --- Save main heatmap figure ---
    base, _ = os.path.splitext(outfile)
    png_out = base + ".png"
    plt.savefig(png_out, bbox_inches="tight", dpi=300, transparent=False, facecolor="white", format="png")
    print(f"✅ Saved main heatmap figure to {png_out}")
    plt.close(fig)

    # === Save colorbars separately ===
    for cmap, norm, label, tag in [
        (cmap_red, norm_red, label_red, "WT_colorbar"),
        (cmap_blue, norm_blue, label_blue, "Ortho_colorbar")
    ]:
        fig_cb, ax_cb = plt.subplots(figsize=(2, 10))  # tall and slim
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig_cb.colorbar(sm, ax=ax_cb, orientation="vertical", fraction=0.3, aspect=20)
        cbar.set_label(label, fontsize=36, weight='bold', labelpad=40)
        cbar.ax.tick_params(labelsize=24, width=2, length=8)

        fig_cb.patch.set_facecolor("white")
        cb_out = f"{base}_{tag}.png"
        plt.savefig(cb_out, bbox_inches="tight", dpi=300, facecolor="white", transparent=False)
        plt.close(fig_cb)
        print(f"🎨 Saved {cb_out} separately")





# ---------- Main ----------
if __name__ == "__main__":

    OUT_DIR = '/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/UP2_pipeline/figures'
    os.makedirs(OUT_DIR, exist_ok=True)

    amino_acid = ['G','A','V','L','I','M','S','T','C','P','N',
                  'Q','F','Y','W','H','K','R','D','E']

    thresholds = {
        "ddG_interface": 0.4,
        "ddG_complex":   0.4,
        "ddG_receptor":  0.05,
    }

    df_wt    = load_sc_file("/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/UP2_pipeline/score_files/wtreceptorup2ligand_4ref.sc")
    df_ortho = load_sc_file("/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/UP2_pipeline/score_files/up2receptorup2ligand_4ref.sc")

    # Sort residues by chain and residue number 
    residues_chainpos = sorted(
        [r for r in df_wt["residue"].unique() if r != "WT"],
        key=lambda x: (x[0], int(re.search(r"\d+", x).group()))
    )

    residues_A = [r for r in residues_chainpos if r.startswith("A")]
    residues_B = [r for r in residues_chainpos if r.startswith("B")]

    # Link row and columns together 
    col_idx_A = {r: j for j, r in enumerate(residues_A)}
    col_idx_B = {r: j for j, r in enumerate(residues_B)}
    row_idx   = {aa: i for i, aa in enumerate(amino_acid)}

    # Build list of wt labels for heatmap x axis
    _, wt_labels_A = get_wt_map_and_labels("/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/UP2_pipeline/input/5repeats_5_wt/relaxed_wtreceptorup2ligand_4.pdb", residues_A)
    _, wt_labels_B = get_wt_map_and_labels("/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/UP2_pipeline/input/5repeats_5_wt/relaxed_wtreceptorup2ligand_4.pdb", residues_B)

    # --- Interface Heatmap---
    wt_array_A    = build_ddg_array(df_wt,    residues_A, amino_acid, col_idx_A, row_idx, "ddG_interface")
    ortho_array_A = build_ddg_array(df_ortho, residues_A, amino_acid, col_idx_A, row_idx, "ddG_interface")
    wt_array_B    = build_ddg_array(df_wt,    residues_B, amino_acid, col_idx_B, row_idx, "ddG_interface")
    ortho_array_B = build_ddg_array(df_ortho, residues_B, amino_acid, col_idx_B, row_idx, "ddG_interface")

    plot_split_heatmaps(wt_array_A, ortho_array_A, residues_A, wt_labels_A,
                        wt_array_B, ortho_array_B, residues_B, wt_labels_B,
                        amino_acid,
                        energy_term="ddG_interface",
                        outfile=os.path.join(OUT_DIR, "0.4threshold_interface_triangles.png"),
                        label_red="WT Interface ΔΔΔG", label_blue="Ortho Interface ΔΔΔG", ablation_threshold=thresholds["ddG_interface"])
    '''
    # --- Complex Heatmap ---
    wt_complex_A    = build_ddg_array(df_wt,    residues_A, amino_acid, col_idx_A, row_idx, "ddG_complex")
    ortho_complex_A = build_ddg_array(df_ortho, residues_A, amino_acid, col_idx_A, row_idx, "ddG_complex")
    wt_complex_B    = build_ddg_array(df_wt,    residues_B, amino_acid, col_idx_B, row_idx, "ddG_complex")
    ortho_complex_B = build_ddg_array(df_ortho, residues_B, amino_acid, col_idx_B, row_idx, "ddG_complex")

    plot_split_heatmaps(wt_complex_A, ortho_complex_A, residues_A, wt_labels_A,
                        wt_complex_B, ortho_complex_B, residues_B, wt_labels_B,
                        amino_acid,
                        energy_term="ddG_complex",
                        outfile=os.path.join(OUT_DIR, "raw_complex_triangles.png"),
                        label_red="WT Complex ΔΔG", label_blue="Ortho Complex ΔΔG", ablation_threshold=thresholds["ddG_complex"])

    # --- Receptor Heatmap---
    wt_receptor_A    = build_ddg_array(df_wt,    residues_A, amino_acid, col_idx_A, row_idx, "ddG_receptor")
    ortho_receptor_A = build_ddg_array(df_ortho, residues_A, amino_acid, col_idx_A, row_idx, "ddG_receptor")
    wt_receptor_B    = build_ddg_array(df_wt,    residues_B, amino_acid, col_idx_B, row_idx, "ddG_receptor")
    ortho_receptor_B = build_ddg_array(df_ortho, residues_B, amino_acid, col_idx_B, row_idx, "ddG_receptor")

    plot_split_heatmaps(wt_receptor_A, ortho_receptor_A, residues_A, wt_labels_A,
                        wt_receptor_B, ortho_receptor_B, residues_B, wt_labels_B,
                        amino_acid,
                        energy_term="ddG_receptor",
                        outfile=os.path.join(OUT_DIR, "raw_receptor_triangles.png"),
                        label_red="WT Receptor ΔΔG", label_blue="Ortho Receptor ΔΔG", ablation_threshold=thresholds["ddG_receptor"])
    '''