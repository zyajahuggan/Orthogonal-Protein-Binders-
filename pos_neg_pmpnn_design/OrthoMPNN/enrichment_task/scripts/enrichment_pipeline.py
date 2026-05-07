#! /usr/bin/env python3
"""
Automated iterative MPNN + Rosetta enrichment pipeline.

Each round:
  1. Run ProteinMPNN on input pdbs (round 1: abcd_complex, subsequent rounds: passing pdbs)
  2. Parse .fa output -> apply mutations to ABCD complex -> save mutant pdbs
  3. Relax mutant pdbs (nstruct replicates each)
  4. Score AB_C interface -> filter ddG < threshold
  5. Copy passing rep1 pdbs -> input for next round

Stops after max_rounds or if 0 designs pass in a round.

NOTE: Before running, extract the 4-chain ABCD structure from the 8-chain file:
  grep "^ATOM\|^HETATM" triple_renumbered.pdb | awk '$5=="A"||$5=="B"||$5=="C"||$5=="D"' > abcd_complex.pdb
  echo "END" >> abcd_complex.pdb
  mv abcd_complex.pdb <base_dir>/input/abcd_complex.pdb
  mkdir -p <base_dir>/input/abcd_input
  cp <base_dir>/input/abcd_complex.pdb <base_dir>/input/abcd_input/
"""
import os
import re
import sys
import shutil
import subprocess
import pandas as pd
from collections import defaultdict
from multiprocessing import Pool

import pyrosetta
from pyrosetta import pose_from_pdb, create_score_function
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.constraint_movers import ClearConstraintsMover
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects


# ============================================================
# CONFIG
# ============================================================
BASE = "/home/zhuggan1/scr4_jgray21/zhuggan1/Orthogonal-Protein-Binders-/pos_neg_pmpnn_design"

CONFIG = dict(
    # Paths
    base_dir        = f"{BASE}/OrthoMPNN/enrichment_task",

    # 4-chain ABCD structure for mutation application and MPNN input
    abcd_pdb        = f"{BASE}/OrthoMPNN/enrichment_task/input/abcd_complex.pdb",

    # Folder containing just abcd_complex.pdb for MPNN round 1 input
    initial_input = f"{BASE}/OrthoMPNN/enrichment_task/input",

    relax_xml       = f"{BASE}/OrthoMPNN/Rosetta/pipelines/reference_structures.xml",

    # ProteinMPNN paths
    mpnn_script     = f"{BASE}/ProteinMPNN/protein_mpnn_run.py",
    mpnn_parse      = f"{BASE}/ProteinMPNN/helper_scripts/parse_multiple_chains.py",
    mpnn_assign     = f"{BASE}/ProteinMPNN/helper_scripts/assign_fixed_chains.py",
    mpnn_tie        = f"{BASE}/OrthoMPNN/enrichment_task/input/make_pos_neg_tie_json.py",
    mpnn_fixed      = f"{BASE}/OrthoMPNN/enrichment_task/input/fixed_positions.py",
    interface_json  = f"{BASE}/OrthoMPNN/enrichment_task/input/interface_residues.json",
    chain_map_json  = f"{BASE}/OrthoMPNN/enrichment_task/input/chain_map.json",
    tied_pairs_json = f"{BASE}/OrthoMPNN/enrichment_task/input/orthogonality_pairs.json",

    # MPNN params
    chains_to_design = "A B",
    num_seq          = 100,
    sampling_temp    = 0.1,

    # Rosetta params
    nstruct          = 5,
    ref_dG           = -114.955,  # avg_dG_interface from reference_scores.csv

    # Pipeline params
    max_rounds       = 5,
    ddg_threshold    = -1.0,
    n_workers        = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
)


# ============================================================
# Amino acid maps
# ============================================================
AA1_TO_AA3 = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
    'E': 'GLU', 'Q': 'GLN', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
    'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
    'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL',
}
AA3_TO_AA1 = {v: k for k, v in AA1_TO_AA3.items()}


# ============================================================
# STEP 1 — Run ProteinMPNN
# ============================================================
def run_mpnn(input_pdb_dir, round_dir, cfg):
    print(f"\n[MPNN] Running on {input_pdb_dir}")
    parsed   = os.path.join(round_dir, "parsed_pdb.jsonl")
    assigned = os.path.join(round_dir, "assigned_chains.jsonl")
    tied     = os.path.join(round_dir, "tied_pos.jsonl")
    fixed    = os.path.join(round_dir, "fixed_positions.jsonl")
    py       = sys.executable

    subprocess.run([py, cfg["mpnn_parse"],
                    "--input_path", input_pdb_dir,
                    "--output_path", parsed], check=True)

    subprocess.run([py, cfg["mpnn_assign"],
                    "--input_path", parsed,
                    "--output_path", assigned,
                    "--chain_list", cfg["chains_to_design"]], check=True)

    subprocess.run([py, cfg["mpnn_tie"],
                    "--input_path", parsed,
                    "--output_path", tied], check=True)

    subprocess.run([py, cfg["mpnn_fixed"],
                    "--input_path", parsed,
                    "--output_path", fixed], check=True)

    subprocess.run([
        py, cfg["mpnn_script"],
        "--jsonl_path",              parsed,
        "--chain_id_jsonl",          assigned,
        "--fixed_positions_jsonl",   fixed,
        "--tied_positions_jsonl",    tied,
        "--out_folder",              round_dir,
        "--conditional_probs_only",  "0",
        "--num_seq_per_target",      str(cfg["num_seq"]),
        "--sampling_temp",           str(cfg["sampling_temp"]),
        "--interface_residues_json", cfg["interface_json"],
        "--chain_map_json",          cfg["chain_map_json"],
        "--tied_pairs_json",         cfg["tied_pairs_json"],
        "--batch_size",              "1",
    ], check=True)

    seqs_dir = os.path.join(round_dir, "seqs")
    fa_files = [f for f in os.listdir(seqs_dir) if f.endswith(".fa")]
    if not fa_files:
        raise FileNotFoundError(f"No .fa files found in {seqs_dir}")
    return os.path.join(seqs_dir, fa_files[0])


# ============================================================
# STEP 2 — Parse .fa and apply mutations
# ============================================================
def parse_fa(fa_path):
    samples = []
    wt_A = wt_B = None
    with open(fa_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    first = True
    while i < len(lines):
        if lines[i].startswith('>'):
            header   = lines[i]
            seq_line = lines[i + 1] if i + 1 < len(lines) else ""
            i += 2
            parts = seq_line.split('/')
            seq_A = parts[0] if len(parts) > 0 else ""
            seq_B = parts[1] if len(parts) > 1 else ""
            if first:
                wt_A, wt_B = seq_A, seq_B
                first = False
            else:
                sample_match = re.search(r'sample=(\d+)', header)
                score_match  = re.search(r'score=([\d.]+)', header)
                samples.append({
                    'sample': int(sample_match.group(1)) if sample_match else -1,
                    'score':  float(score_match.group(1)) if score_match else None,
                    'seq_A':  seq_A,
                    'seq_B':  seq_B,
                })
        else:
            i += 1
    return wt_A, wt_B, samples


def get_mutations(wt_seq, mut_seq, chain):
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            mutations.append((chain, i + 1, AA1_TO_AA3[mut_aa]))
    return mutations


def mutation_label(mutations):
    return '_'.join(f"{c}{r}{AA3_TO_AA1[a]}" for c, r, a in mutations)


def apply_mutations_to_pose(pose, mutations):
    for chain, resnum, aa3 in mutations:
        pose_res = pose.pdb_info().pdb2pose(chain, resnum)
        if pose_res == 0:
            raise ValueError(f"Residue {chain}{resnum} not found.")
        MutateResidue(pose_res, aa3).apply(pose)


def build_mutant_pdbs(fa_path, abcd_pdb, mutant_pdb_dir):
    print(f"\n[MUTATE] Parsing {fa_path}")
    os.makedirs(mutant_pdb_dir, exist_ok=True)

    wt_A, wt_B, samples = parse_fa(fa_path)
    profile_to_mutations = {}
    profile_to_samples   = defaultdict(list)

    for s in samples:
        muts = get_mutations(wt_A, s['seq_A'], 'A') + get_mutations(wt_B, s['seq_B'], 'B')
        key  = mutation_label(muts) if muts else "WT"
        profile_to_mutations[key] = muts
        profile_to_samples[key].append(s['sample'])

    saved = 0
    for key, muts in profile_to_mutations.items():
        if key == "WT":
            continue
        try:
            pose = pose_from_pdb(abcd_pdb)
            apply_mutations_to_pose(pose, muts)
            pose.dump_pdb(os.path.join(mutant_pdb_dir, f"{key}.pdb"))
            saved += 1
        except Exception as e:
            print(f"  ERROR on {key}: {e}")

    print(f"[MUTATE] Saved {saved} mutant pdb(s) to {mutant_pdb_dir}")


# ============================================================
# STEP 3 — Relax mutant pdbs
# ============================================================
GLOBAL_FRELAX = None


def init_relax_worker(xml_file):
    pyrosetta.init("-mute all", silent=True)
    global GLOBAL_FRELAX
    xmlobj = XmlObjects.create_from_file(xml_file)
    GLOBAL_FRELAX = xmlobj.get_mover("FastRelax")


def relax_one(args):
    pdb_path, out_path = args
    try:
        pose = pose_from_pdb(pdb_path)
        GLOBAL_FRELAX.apply(pose)
        pose.dump_pdb(out_path)
        print(f"  Relaxed: {os.path.basename(out_path)}")
        return out_path
    except Exception as e:
        print(f"  ERROR relaxing {pdb_path}: {e}")
        return None


def relax_mutants(mutant_pdb_dir, relaxed_dir, relax_xml, nstruct, n_workers):
    print(f"\n[RELAX] Relaxing pdbs in {mutant_pdb_dir}")
    os.makedirs(relaxed_dir, exist_ok=True)

    jobs = []
    for fname in os.listdir(mutant_pdb_dir):
        if not fname.endswith(".pdb"):
            continue
        mut_key = fname.replace(".pdb", "")
        sub_dir = os.path.join(relaxed_dir, mut_key)
        os.makedirs(sub_dir, exist_ok=True)
        src = os.path.join(mutant_pdb_dir, fname)
        for i in range(1, nstruct + 1):
            out = os.path.join(sub_dir, f"relaxed_{i}.pdb")
            jobs.append((src, out))

    with Pool(processes=n_workers,
              initializer=init_relax_worker,
              initargs=(relax_xml,)) as pool:
        pool.map(relax_one, jobs)

    print(f"[RELAX] Done.")


# ============================================================
# STEP 4 — Score and filter
# ============================================================
def init_score_worker():
    pyrosetta.init("-mute all")


def score_one(args):
    pdb_path, mutation = args
    try:
        pose = pose_from_pdb(pdb_path)
        ClearConstraintsMover().apply(pose)
        scorefxn = create_score_function("ref2015")
        iam = InterfaceAnalyzerMover("AB_C")
        iam.set_scorefunction(scorefxn)
        iam.set_pack_separated(True)
        iam.set_pack_rounds(5)
        iam.apply(pose)
        dG      = iam.get_interface_dG()
        fname   = os.path.basename(pdb_path)
        rep     = re.search(r"_(\d+)\.pdb$", fname)
        rep_num = int(rep.group(1)) if rep else -1
        return {"mutation": mutation, "rep": rep_num, "description": fname, "dG_interface": dG}
    except Exception as e:
        print(f"  Skipping {pdb_path}: {e}")
        return None


def score_and_filter(relaxed_dir, ref_dG, out_csv, passing_dir, ddg_threshold, n_workers):
    print(f"\n[SCORE] Scoring pdbs in {relaxed_dir}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(passing_dir, exist_ok=True)

    jobs = []
    for mutation in sorted(os.listdir(relaxed_dir)):
        subdir = os.path.join(relaxed_dir, mutation)
        if not os.path.isdir(subdir):
            continue
        for fname in os.listdir(subdir):
            if fname.endswith(".pdb"):
                jobs.append((os.path.join(subdir, fname), mutation))

    pyrosetta.init("-mute all", silent=True)
    with Pool(processes=n_workers, initializer=init_score_worker) as pool:
        raw = pool.map(score_one, jobs)

    records = [r for r in raw if r is not None]
    df = pd.DataFrame(records)

    averaged = (
        df.groupby("mutation")["dG_interface"]
        .mean().reset_index()
        .rename(columns={"dG_interface": "avg_dG_interface"})
    )
    averaged["ddG_interface"] = averaged["avg_dG_interface"] - ref_dG
    rep_counts = (df.groupby("mutation")["rep"]
                  .count().reset_index()
                  .rename(columns={"rep": "n_reps"}))
    rep1_df = (df[df["rep"] == 1][["mutation", "description"]]
               .drop_duplicates("mutation"))
    averaged = (averaged
                .merge(rep_counts, on="mutation")
                .merge(rep1_df, on="mutation", how="left")
                .sort_values("ddG_interface"))
    averaged.to_csv(out_csv, index=False)

    print(averaged[["mutation", "avg_dG_interface", "ddG_interface", "n_reps"]].to_string(index=False))

    passing = averaged[averaged["ddG_interface"] < ddg_threshold]
    print(f"\n{len(passing)}/{len(averaged)} mutation(s) pass ddG < {ddg_threshold}")

    copied = 0
    for _, row in passing.iterrows():
        if pd.isna(row["description"]):
            continue
        src = os.path.join(relaxed_dir, row["mutation"], row["description"])
        dst = os.path.join(passing_dir, f"{row['mutation']}_rep1.pdb")
        if os.path.exists(src):
            shutil.copy(src, dst)
            copied += 1
            print(f"  PASS: {row['mutation']}  ddG={row['ddG_interface']:.3f}")
        else:
            print(f"  WARNING: {src} not found.")

    print(f"[SCORE] {copied} passing pdb(s) copied to {passing_dir}")
    return copied


# ============================================================
# Main loop
# ============================================================
def main():
    cfg      = CONFIG
    base_dir = cfg["base_dir"]

    pyrosetta.init("-mute all", silent=True)

    # Round 1 starts from the abcd_input folder (contains just abcd_complex.pdb)
    current_input_dir = cfg["initial_input"]

    for rnd in range(1, cfg["max_rounds"] + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {rnd}")
        print(f"{'='*60}")

        round_dir   = os.path.join(base_dir, "scripts",         f"r{rnd}")
        mutant_dir  = os.path.join(base_dir, "mutant_pdbs",     f"r{rnd}")
        relaxed_dir = os.path.join(base_dir, "rosetta_outputs", f"r{rnd}")
        scores_dir  = os.path.join(base_dir, "rosetta_outputs", "scores")
        passing_dir = os.path.join(base_dir, "rosetta_outputs", f"passing_r{rnd}")
        out_csv     = os.path.join(scores_dir, f"r{rnd}_mutant_scores.csv")

        for d in [round_dir, mutant_dir, relaxed_dir, scores_dir, passing_dir]:
            os.makedirs(d, exist_ok=True)

        # Step 1 — MPNN
        fa_path = run_mpnn(current_input_dir, round_dir, cfg)

        # Step 2 — Apply mutations to ABCD complex
        build_mutant_pdbs(fa_path, cfg["abcd_pdb"], mutant_dir)

        # Step 3 — Relax
        relax_mutants(mutant_dir, relaxed_dir, cfg["relax_xml"], cfg["nstruct"], cfg["n_workers"])

        # Step 4 — Score + filter
        n_passing = score_and_filter(
            relaxed_dir, cfg["ref_dG"], out_csv, passing_dir,
            cfg["ddg_threshold"], cfg["n_workers"]
        )

        if n_passing == 0:
            print(f"\n[PIPELINE] 0 designs passed in round {rnd}. Stopping early.")
            break

        current_input_dir = passing_dir
        print(f"\n[PIPELINE] Round {rnd} complete. {n_passing} design(s) passing to round {rnd + 1}.")

    print("\n[PIPELINE] Enrichment complete.")


if __name__ == "__main__":
    main()