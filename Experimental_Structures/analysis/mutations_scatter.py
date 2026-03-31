import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
import sys
from collections import defaultdict


def extract_mut_tag(name):
    muts = re.findall(r"[A-Z]\d+[A-Z]", str(name))
    return "_".join(muts) if muts else None


def load_and_merge(wt_csv, ortho_csv):
    wt = pd.read_csv(wt_csv)
    ortho = pd.read_csv(ortho_csv)

    # use the correct WT name column
    wt_name_col = "mutation" if "mutation" in wt.columns else "description"
    ortho_name_col = "mutation"

    wt["mut_tag"] = wt[wt_name_col].apply(extract_mut_tag)
    ortho["mut_tag"] = ortho[ortho_name_col].apply(extract_mut_tag)

    merged = pd.merge(
        wt[["mut_tag", wt_name_col, "ddG_complex"]],
        ortho[["mut_tag", ortho_name_col, "ddG_complex"]],
        on="mut_tag",
        suffixes=("_wt", "_ortho")
    )

    # rename for consistency
    merged = merged.rename(columns={
        wt_name_col: "mutation_wt",
        ortho_name_col: "mutation_ortho"
    })

    # if WT has replicates, average them so you get one point per mutation
    merged = (
        merged.groupby(["mut_tag", "mutation_ortho"], as_index=False)
        .agg({
            "ddG_complex_wt": "mean",
            "ddG_complex_ortho": "first"
        })
    )

    print("Merged rows:", len(merged))
    print("Unique mutations:", merged["mut_tag"].nunique())
    print(merged[["mut_tag", "ddG_complex_wt", "ddG_complex_ortho"]])

    return merged


def plot_ddg_scatter_colored(merged, outdir="../../figures"):
    os.makedirs(outdir, exist_ok=True)

    x = merged["ddG_complex_wt"].values
    y = merged["ddG_complex_ortho"].values
    labels = [
        "_".join(m.split("_")[-7:])
        for m in merged["mutation_ortho"]
    ]

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

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])
    ax = fig.add_subplot(gs[0])
    ax.set_box_aspect(1)

    # duplicate-aware jitter
    coord_counts = defaultdict(int)

    for i in range(len(labels)):
        xi, yi = x[i], y[i]

        key = (round(float(xi), 3), round(float(yi), 3))
        count = coord_counts[key]

        if count == 0:
            xi_plot, yi_plot = xi, yi
        else:
            angle = count * (np.pi / 4)
            radius = 0.12 * count
            xi_plot = xi + radius * np.cos(angle)
            yi_plot = yi + radius * np.sin(angle)

        coord_counts[key] += 1

        ax.scatter(
            xi_plot, yi_plot,
            color=colors[i % len(colors)],
            s=90,
            edgecolor="black",
            linewidth=0.6,
            zorder=3
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.2)
    ax.axvline(0, color="black", linestyle="-", linewidth=1.2)

    ax.axhline(-1, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(1, color="red", linestyle="--", linewidth=1.2)

    ax.set_xlabel("WTR:OrthoL ΔΔΔG Complex (REU)", fontsize=12)
    ax.set_ylabel("OrthoR:OrthoL ΔΔΔG Complex (REU)", fontsize=12)
    ax.set_title("WTR:OrthoL vs OrthoR:OrthoL ΔΔΔG Complex", fontsize=14)
    ax.grid(alpha=0.3)

    # automatic limits so all points appear
    xpad = max(0.8, 0.08 * (np.max(x) - np.min(x) if len(x) > 1 else 1))
    ypad = max(0.8, 0.08 * (np.max(y) - np.min(y) if len(y) > 1 else 1))

    ax.set_xlim(np.min(x) - xpad, np.max(x) + xpad)
    ax.set_ylim(np.min(y) - ypad, np.max(y) + ypad)

    ax.set_aspect("equal", adjustable="box")

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

    outpath = os.path.join(outdir, "global_complex_relaxed_scatter.png")
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ Saved to {outpath}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <wt_avg.csv> <ortho_avg.csv>")
        sys.exit(1)

    wt_csv = sys.argv[1]
    ortho_csv = sys.argv[2]

    merged = load_and_merge(wt_csv, ortho_csv)
    plot_ddg_scatter_colored(merged)