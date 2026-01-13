#! /usr/bin/env python3
import pandas as pd
import sys
import os

def convert_sc_to_csv(sc_file, outdir="/scratch/jgray21/zyhuggan/Ortho_BB/csv_files"):
    """
    Converts a Rosetta .sc scorefile to a clean .csv file.
    Outputs the CSV to the specified directory (default: 'csvs/').
    """
    # --- Ensure output directory exists ---
    os.makedirs(outdir, exist_ok=True)

    # --- Read all lines ---
    with open(sc_file) as f:
        lines = f.readlines()

    # --- Find header and data lines ---
    header_line = [l for l in lines if l.startswith("SCORE:")][0]
    header = header_line.strip().split()[1:]  # remove 'SCORE:'
    data = [l.strip().split()[1:] for l in lines if l.startswith("SCORE:")][1:]

    # --- Convert to DataFrame ---
    df = pd.DataFrame(data, columns=header)
    df = df.apply(pd.to_numeric, errors="ignore")

    # --- Output path ---
    base = os.path.basename(sc_file)
    csv_name = os.path.splitext(base)[0] + ".csv"
    csv_path = os.path.join(outdir, csv_name)

    # --- Write to CSV ---
    df.to_csv(csv_path, index=False)
    print(f"✅ {sc_file} → {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_sc_to_csv.py <file1.sc> [file2.sc] ...")
        sys.exit(1)

    for sc_file in sys.argv[1:]:
        convert_sc_to_csv(sc_file)
