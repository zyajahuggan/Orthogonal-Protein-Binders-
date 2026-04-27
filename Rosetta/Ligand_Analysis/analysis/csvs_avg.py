#! /usr/bin/env python3
import pandas as pd
import os
import re
import sys

def average_replicates(csv_file, outdir="/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/interface_relax/csv_files"):
    """
    For each unique mutation (e.g., A_81_R), average across its 5 replicates.
    Example filenames: relaxed_wt_1_A_81_R_rep1.pdb ... rep5.pdb
    """
    df = pd.read_csv(csv_file)
    

    # Extract mutation identifier (everything before _rep#)
    df["mutation"] = df["description"].str.extract(r"(.+)_rep\d+\.pdb")

    # Identify numeric columns for averaging
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Average numeric values across all replicates for each mutation
    averaged = (
        df.groupby("mutation")[numeric_cols]
        .mean()
        .reset_index()
    )

    # Keep one representative description (e.g. the first rep)
    first_desc = (
        df.groupby("mutation")["description"]
        .first()
        .reset_index()
    )

    # Merge averaged data and representative description
    result = pd.merge(averaged, first_desc, on="mutation", how="left")

    result = result.iloc[:, :-1]

    # Output path
    base = os.path.basename(csv_file)
    out_name = os.path.splitext(base)[0] + "_avg.csv"
    out_path = os.path.join(outdir, out_name)

    os.makedirs(outdir, exist_ok=True)
    result.to_csv(out_path, index=False)

    print(f"✅ Averaged replicates saved to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python average_mutation_replicates.py <file1.csv> [file2.csv] ...")
        sys.exit(1)

    for file in sys.argv[1:]:
        average_replicates(file)
