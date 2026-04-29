#! /usr/bin/env python3
"""
Scores the AB_C interface for all mutant replicate pdbs organized as:
    r1/<mutation_name>/relaxed_reference_1.pdb
    r1/<mutation_name>/relaxed_reference_2.pdb
    ...

Averages replicates per mutation (subfolder), computes ddG vs a
pre-calculated reference avg_dG_interface, filters on ddG < threshold,
and saves results to csv.
"""
import os
import re
import shutil
import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from multiprocessing import Pool


# -----------------------------
# Worker initializer
# -----------------------------
def init_worker():
    pyrosetta.init("-mute all")


# -----------------------------
# Core scoring routine — AB_C interface
# -----------------------------
def calculate_interface_score(pdb_path):
    pose = pose_from_pdb(pdb_path)
    ClearConstraintsMover().apply(pose)

    scorefxn = create_score_function("ref2015")

    iam = InterfaceAnalyzerMover("AB_C")
    iam.set_scorefunction(scorefxn)
    iam.set_pack_separated(True)
    iam.set_pack_rounds(5)
    iam.apply(pose)

    return iam.get_interface_dG()


# -----------------------------
# Score one pdb
# -----------------------------
def score_one(args):
    pdb_path, mutation = args
    try:
        dG      = calculate_interface_score(pdb_path)
        fname   = os.path.basename(pdb_path)
        rep     = re.search(r"_(\d+)\.pdb$", fname)
        rep_num = int(rep.group(1)) if rep else -1
        print(f"  Scored {mutation}//{fname}: dG_interface = {dG:.3f}")
        return {
            "description":  fname,
            "mutation":     mutation,
            "rep":          rep_num,
            "dG_interface": dG,
        }
    except Exception as e:
        print(f"  Skipping {pdb_path}: {e}")
        return None


# -----------------------------
# Main
# -----------------------------
def main(
    r1_dir,             # top-level folder containing one subfolder per mutation
    ref_dG,             # avg_dG_interface from reference scoring step
    out_csv,            # output csv path
    out_pdb_dir,        # directory to copy passing rep1 pdbs into
    ddg_threshold=-1.0, # filter: avg ddG must be below this
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(out_pdb_dir, exist_ok=True)

    pyrosetta.init("-mute all", silent=True)

    # Walk subfolders — each subfolder is one mutation
    jobs = []
    for mutation in sorted(os.listdir(r1_dir)):
        subdir = os.path.join(r1_dir, mutation)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.endswith(".pdb"):
                jobs.append((os.path.join(subdir, fname), mutation))

    print(f"Found {len(jobs)} replicate pdb(s) across {len(set(j[1] for j in jobs))} mutation(s).")
    print(f"Using reference avg_dG_interface = {ref_dG:.3f}\n")

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Using {n_workers} worker(s)")

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        raw_results = pool.map(score_one, jobs)

    records = [r for r in raw_results if r is not None]
    df = pd.DataFrame(records)

    # Average replicates per mutation
    averaged = (
        df.groupby("mutation")["dG_interface"]
        .mean()
        .reset_index()
        .rename(columns={"dG_interface": "avg_dG_interface"})
    )

    # Compute ddG vs reference
    averaged["ddG_interface"] = averaged["avg_dG_interface"] - ref_dG

    # Attach rep count
    rep_counts = (
        df.groupby("mutation")["rep"]
        .count()
        .reset_index()
        .rename(columns={"rep": "n_reps"})
    )
    averaged = averaged.merge(rep_counts, on="mutation")

    # Attach rep1 pdb path for copying
    rep1_df = (
        df[df["rep"] == 1][["mutation", "description"]]
        .drop_duplicates("mutation")
    )
    averaged = averaged.merge(rep1_df, on="mutation", how="left")

    # Sort by ddG
    averaged = averaged.sort_values("ddG_interface")

    # Save full csv
    averaged.to_csv(out_csv, index=False)
    print(f"\nSaved scores to: {out_csv}")
    print(averaged[["mutation", "avg_dG_interface", "ddG_interface", "n_reps"]].to_string(index=False))

    # Filter and copy passing rep1 pdbs
    passing = averaged[averaged["ddG_interface"] < ddg_threshold]
    print(f"\n{len(passing)}/{len(averaged)} mutation(s) pass ddG < {ddg_threshold}")

    copied = 0
    for _, row in passing.iterrows():
        rep1_fname = row["description"]
        if pd.isna(rep1_fname):
            print(f"  WARNING: no rep1 found for {row['mutation']}, skipping.")
            continue
        src = os.path.join(r1_dir, row["mutation"], rep1_fname)
        dst = os.path.join(out_pdb_dir, f"{row['mutation']}_rep1.pdb")
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
            print(f"  PASS: {row['mutation']}  ddG={row['ddG_interface']:.3f}")
        else:
            print(f"  WARNING: {src} not found.")

    print(f"\nDone. {copied} passing pdb(s) copied to: {out_pdb_dir}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    main(
        r1_dir        = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs/r1",
        ref_dG        = -114.8745630944571,  # <-- replace with AVERAGE value from reference_scores.csv
        out_csv       = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs/scores/r1_mutant_scores.csv",
        out_pdb_dir   = "/scratch/jgray21/zyhuggan/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design/OrthoMPNN/enrichment_task/rosetta_outputs/passing_r1",
        ddg_threshold = -1.0,
    )