import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import sys


def extract_mut_tag(name):
    muts = re.findall(r"[A-Z]\d+[A-Z]", name)
    return "_".join(muts) if muts else None


def load_and_merge(wt_csv, ortho_csv):
    wt = pd.read_csv(wt_csv)
    ortho = pd.read_csv(ortho_csv)

    wt["mut_tag"] = wt["mutation"].apply(extract_mut_tag)
    ortho["mut_tag"] = ortho["mutation"].apply(extract_mut_tag)

    merged = pd.merge(
        wt[["mut_tag", "mutation", "ddG_interface"]],
        ortho[["mut_tag", "mutation", "ddG_interface"]],
        on="mut_tag",
        suffixes=("_wt", "_ortho")
    )

    print("Merged rows:", len(merged))
    print("Unique mutations:", merged["mut_tag"].nunique())

    return merged


def plot_ddg_scatter_colored(merged, outdir="../../figures"):

    os.makedirs(outdir, exist_ok=True)

    x = merged["ddG_interface_wt"].values
    y = merged["ddG_interface_ortho"].values
    labels = [
        "_".join(m.split("_")[-7:])
        for m in merged["mutation_ortho"]
    ]

    # --- Colors ---
    cmap = plt.cm.get_cmap("tab20", len(labels))
    colors = [
        "#0072B2",  # blue
        "#D55E00",  # red-orange
        "#009E73",  # green
        "#CC79A7",  # purple
        "#56B4E9",  # light blue
        "#E69F00",  # orange
        "#000000",  # black
        "#7F7F7F",  # gray
    ]

    fig = plt.figure(figsize=(8,10))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax = fig.add_subplot(gs[0])

    # --- Scatter ---
    for i in range(len(labels)):
        ax.scatter(
            x[i], y[i],
            color=colors[i],
            s=80,
            edgecolor="black",
            linewidth=0.6
        )

    # Axes
    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)

    ax.axhline(-1, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(1, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel("WTR:OrthoL ΔΔΔG Interface (REU)", fontsize=12)
    ax.set_ylabel("OrthoR:OrthoL ΔΔΔG Interface (REU)", fontsize=12)
    ax.set_title("WTR:OrthoL vs OrthoR:OrthoL ΔΔΔG Interface ", fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 10)
    ax.set_aspect('equal', adjustable='box')

    # --- Mutation list below ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")

    n_cols = 2
    rows = int(np.ceil(len(labels) / n_cols))

    for i, (lab, col) in enumerate(zip(labels, colors)):
        r = i % rows
        c = i // rows
        ax2.text(
            0.02 + c * 0.48,
            0.95 - r * 0.12,
            lab,
            color=col,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="top",
            wrap=True
        )

    plt.tight_layout()

    outpath = os.path.join(outdir, "colored_ddg_scatter.png")
    plt.savefig(outpath, dpi=300)
    plt.show()

    print(f"✅ Saved to {outpath}")


# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <wt_avg.csv> <ortho_avg.csv>")
        sys.exit(1)

    wt_csv = sys.argv[1]
    ortho_csv = sys.argv[2]

    merged = load_and_merge(wt_csv, ortho_csv)
    plot_ddg_scatter_colored(merged)