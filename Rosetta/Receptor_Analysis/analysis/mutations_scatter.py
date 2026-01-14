#! /usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import sys

#Quadrant counts:
#Q1 (x>0, y>0): 599
#Q2 (x<0, y>0): 91
#Q3 (x<0, y<0): 121
#Q4 (x>0, y<0): 142

def extract_mut_tag(name):
    """Extract mutation ID (e.g. A_21_C from relaxed_wt_1_A_21_C)."""
    m = re.search(r"([A-Z]_\d+_[A-Z])", name)
    return m.group(1) if m else None

def plot_ddg_scatter(wt_csv, ortho_csv, outdir="../../figures"):
    os.makedirs(outdir, exist_ok=True)

    # --- Load data ---
    wt = pd.read_csv(wt_csv)
    ortho = pd.read_csv(ortho_csv)

    # --- Normalize mutation tags ---
    wt["mut_tag"] = wt["mutation"].apply(extract_mut_tag)
    ortho["mut_tag"] = ortho["mutation"].apply(extract_mut_tag)

    # --- Merge on mut_tag ---
    merged = pd.merge(
        wt[["mut_tag", "ddG_interface"]],
        ortho[["mut_tag", "ddG_interface"]],
        on="mut_tag",
        suffixes=("_wt", "_ortho")
    )
    x = merged["ddG_interface_wt"]
    y = merged["ddG_interface_ortho"]

    q1 = ((x > 0) & (y > 0)).sum()
    q2 = ((x < 0) & (y > 0)).sum()
    q3 = ((x < 0) & (y < 0)).sum()
    q4 = ((x > 0) & (y < 0)).sum()

    print("\nQuadrant counts:")
    print(f"Q1 (x>0, y>0): {q1}")
    print(f"Q2 (x<0, y>0): {q2}")
    print(f"Q3 (x<0, y<0): {q3}")
    print(f"Q4 (x>0, y<0): {q4}")

    total = len(merged)
    on_x_axis = ((x == 0) & (y != 0)).sum()
    on_y_axis = ((y == 0) & (x != 0)).sum()
    on_origin = ((x == 0) & (y == 0)).sum()
    nan_count = x.isna().sum() + y.isna().sum()  # just in case

    print("\nQuadrant counts:")
    print(f"Q1 (x>0, y>0): {q1}")
    print(f"Q2 (x<0, y>0): {q2}")
    print(f"Q3 (x<0, y<0): {q3}")
    print(f"Q4 (x>0, y<0): {q4}")
    print(f"On x-axis: {on_x_axis}")
    print(f"On y-axis: {on_y_axis}")
    print(f"On origin: {on_origin}")
    print(f"NaN values: {nan_count}")
    print(f"Total points accounted for: {q1+q2+q3+q4+on_x_axis+on_y_axis+on_origin}")
    print(f"Expected total: {total}")

    wt_tags = set(wt["mut_tag"])
    ortho_tags = set(ortho["mut_tag"])

    missing_in_ortho = wt_tags - ortho_tags
    missing_in_wt = ortho_tags - wt_tags

    print(f"Total WT mutations: {len(wt_tags)}")
    print(f"Total Ortho mutations: {len(ortho_tags)}")
    print(f"Mutations missing in Ortho: {len(missing_in_ortho)}")
    print(f"Mutations missing in WT: {len(missing_in_wt)}")

        # --- Identify lower-right quadrant (Q4) ---
    q4_mask = (x > 0) & (y < 0)
    q4_df = merged.loc[q4_mask, ["mut_tag", "ddG_interface_wt", "ddG_interface_ortho"]]

    # Rank by strongest orthogonality (largest x, most negative y)
    q4_df["orthogonality_score"] = q4_df["ddG_interface_wt"] - q4_df["ddG_interface_ortho"]
    q4_df = q4_df.sort_values("orthogonality_score", ascending=False)

    # Print top hits
    print("\nTop Q4 (Ortho-specific) mutations:")
    print(q4_df.head(20).to_string(index=False))

    # --- Plot ---
    plt.figure(figsize=(7,7))
    plt.scatter(
        merged["ddG_interface_wt"],
        merged["ddG_interface_ortho"],
        color="#6E6E6E",          # medium gray
        s=70,
        edgecolor="black",
        alpha=0.8,
        linewidth=0.5
    )

    # --- Highlight residue 34 mutations ---
    res34_mask = merged["mut_tag"].str.contains(r"_34_", regex=True)
    res34_df = merged[res34_mask]

    plt.scatter(
        res34_df["ddG_interface_wt"],
        res34_df["ddG_interface_ortho"],
        color="crimson",
        s=60,
        edgecolor="black",
        linewidth=0.8,
        label="Residue 34 mutants"
    )

    # Add horizontal (y=0) and vertical (x=0) black dashed lines
    plt.axhline(0, color="black", linestyle="--", linewidth=1.2)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.2)

    # Labels and formatting
    plt.xlabel("WTR:OrthoL ΔΔΔG_rel (REU)", fontsize=12)
    plt.ylabel("OrthoR:OrthoL ΔΔΔG_rel (REU)", fontsize=12)
    plt.title("ΔΔG Comparison", fontsize=14)
    plt.grid(alpha=0.3)
    plt.axis("square")
    plt.xlim(-20,30)

    # --- Save ---
    outpath = os.path.join(outdir, "34highlight_ddg_interface_scatter_gray.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()

    print(f"✅ Scatter plot saved to {outpath}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python ddg_interface_scatter_gray.py <wt_avg.csv> <ortho_avg.csv>")
        sys.exit(1)
    plot_ddg_scatter(sys.argv[1], sys.argv[2])
