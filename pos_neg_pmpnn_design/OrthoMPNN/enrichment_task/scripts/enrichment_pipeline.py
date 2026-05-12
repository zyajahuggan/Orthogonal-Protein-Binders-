#! /usr/bin/env python3
"""
Automated iterative MPNN + Rosetta enrichment pipeline.

Each round:
  1. Run ProteinMPNN on input pdbs
  2. Parse .fa output -> compare each sequence's MPNN score to the WT score
     from that same .fa file (lower = better in MPNN log-prob scoring)
  3. Keep sequences that are better than WT
  4. Apply mutations to ABCD complex -> relax mutant pdbs
  5. Pass relaxed pdbs as input for next round

Stops after max_rounds or if 0 sequences beat WT score in a round.
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
from pyrosetta import pose_from_pdb
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
    abcd_pdb        = f"{BASE}/OrthoMPNN/enrichment_task/input/abcd_complex.pdb",
    initial_input   = f"{BASE}/OrthoMPNN/enrichment_task/input",
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

    # Rosetta params (relax only, no scoring filter)
    nstruct          = 5,

    # Pipeline params
    max_rounds       = 5,
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
# STEP 2 — Parse .fa and filter by MPNN score
# ============================================================
def parse_fa(fa_path):
    """
    Parse .fa file. Returns:
      wt_score  : float  — score of the first (WT/reference) sequence
      wt_A, wt_B: str   — WT sequences for chains A and B
      samples   : list of dicts with keys: sample, score, seq_A, seq_B
    """
    samples  = []
    wt_score = None
    wt_A = wt_B = None

    with open(fa_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    i     = 0
    first = True
    while i < len(lines):
        if lines[i].startswith('>'):
            header   = lines[i]
            seq_line = lines[i + 1] if i + 1 < len(lines) else ""
            i += 2
            parts = seq_line.split('/')
            seq_A = parts[0] if len(parts) > 0 else ""
            seq_B = parts[1] if len(parts) > 1 else ""

            score_match = re.search(r'score=([\d.]+)', header)
            score = float(score_match.group(1)) if score_match else None

            if first:
                # First entry is always the WT/reference sequence
                wt_score = score
                wt_A, wt_B = seq_A, seq_B
                first = False
                print(f"[FILTER] WT MPNN score = {wt_score:.4f}")
            else:
                sample_match = re.search(r'sample=(\d+)', header)
                samples.append({
                    'sample': int(sample_match.group(1)) if sample_match else -1,
                    'score':  score,
                    'seq_A':  seq_A,
                    'seq_B':  seq_B,
                })
        else:
            i += 1

    return wt_score, wt_A, wt_B, samples


def get_mutations(wt_seq, mut_seq, chain):
    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt_seq, mut_seq)):
        if wt_aa != mut_aa:
            mutations.append((chain, i + 1, AA1_TO_AA3[mut_aa]))
    return mutations


def mutation_label(mutations):
    return '_'.join(f"{c}{r}{AA3_TO_AA1[a]}" for c, r, a in mutations)


def filter_and_build_mutants(fa_path, abcd_pdb, mutant_pdb_dir, round_dir):
    """
    Parse .fa, keep sequences with MPNN score < WT score,
    apply mutations to ABCD complex, save mutant PDBs.
    Returns number of passing sequences saved.
    """
    print(f"\n[FILTER] Parsing {fa_path}")
    os.makedirs(mutant_pdb_dir, exist_ok=True)

    wt_score, wt_A, wt_B, samples = parse_fa(fa_path)

    # Filter: lower MPNN score = better (log-prob, less negative = better fit)
    passing = [s for s in samples if s['score'] is not None and s['score'] < wt_score]
    print(f"[FILTER] {len(passing)}/{len(samples)} sequences beat WT score ({wt_score:.4f})")

    if not passing:
        return 0

    # Deduplicate by mutation profile
    profile_to_best = {}
    for s in passing:
        muts = get_mutations(wt_A, s['seq_A'], 'A') + get_mutations(wt_B, s['seq_B'], 'B')
        key  = mutation_label(muts) if muts else "WT"
        if key == "WT":
            continue
        # Keep best (lowest) score per unique mutation profile
        if key not in profile_to_best or s['score'] < profile_to_best[key]['score']:
            profile_to_best[key] = {'score': s['score'], 'muts': muts}

    print(f"[FILTER] {len(profile_to_best)} unique mutation profile(s) after deduplication")

    # Save scores CSV for this round
    scores_csv = os.path.join(round_dir, "mpnn_scores.csv")
    rows = [{"mutation": k, "mpnn_score": v['score'], "delta_score": v['score'] - wt_score}
            for k, v in sorted(profile_to_best.items(), key=lambda x: x[1]['score'])]
    pd.DataFrame(rows).to_csv(scores_csv, index=False)
    print(f"[FILTER] Scores saved to {scores_csv}")

    # Apply mutations and save PDBs
    saved = 0
    for key, info in profile_to_best.items():
        try:
            pose = pose_from_pdb(abcd_pdb)
            for chain, resnum, aa3 in info['muts']:
                pose_res = pose.pdb_info().pdb2pose(chain, resnum)
                if pose_res == 0:
                    raise ValueError(f"Residue {chain}{resnum} not found.")
                MutateResidue(pose_res, aa3).apply(pose)
            out_pdb = os.path.join(mutant_pdb_dir, f"{key}.pdb")
            pose.dump_pdb(out_pdb)
            saved += 1
        except Exception as e:
            print(f"  ERROR on {key}: {e}")

    print(f"[FILTER] Saved {saved} mutant pdb(s) to {mutant_pdb_dir}")
    return saved


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
    """
    Relax each mutant PDB nstruct times. The best (lowest energy) replicate
    per mutant is copied to relaxed_dir root as the representative for next round.
    """
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


def collect_best_relaxed(relaxed_dir, passing_dir, cfg):
    """
    For each mutant, pick rep1 (consistent with original pipeline) and
    copy to passing_dir so it can be fed into the next MPNN round.
    """
    os.makedirs(passing_dir, exist_ok=True)
    scorefxn = None  # use rep1 for consistency; swap to energy-based if desired

    copied = 0
    for mut_key in os.listdir(relaxed_dir):
        sub_dir = os.path.join(relaxed_dir, mut_key)
        if not os.path.isdir(sub_dir):
            continue
        rep1 = os.path.join(sub_dir, "relaxed_1.pdb")
        if os.path.exists(rep1):
            dst = os.path.join(passing_dir, f"{mut_key}.pdb")
            shutil.copy(rep1, dst)
            copied += 1

    print(f"[RELAX] {copied} representative pdb(s) copied to {passing_dir}")
    return copied


# ============================================================
# Main loop
# ============================================================
def main():
    cfg      = CONFIG
    base_dir = cfg["base_dir"]

    pyrosetta.init("-mute all", silent=True)

    current_input_dir = cfg["initial_input"]

    for rnd in range(1, cfg["max_rounds"] + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {rnd}")
        print(f"{'='*60}")

        round_dir   = os.path.join(base_dir, "scripts",         f"r{rnd}")
        mutant_dir  = os.path.join(base_dir, "mutant_pdbs",     f"r{rnd}")
        relaxed_dir = os.path.join(base_dir, "rosetta_outputs", f"r{rnd}")
        passing_dir = os.path.join(base_dir, "rosetta_outputs", f"passing_r{rnd}")

        for d in [round_dir, mutant_dir, relaxed_dir, passing_dir]:
            os.makedirs(d, exist_ok=True)

        # Step 1 — Run MPNN
        fa_path = run_mpnn(current_input_dir, round_dir, cfg)

        # Step 2 — Filter by MPNN score vs WT, build mutant PDBs
        n_passing = filter_and_build_mutants(fa_path, cfg["abcd_pdb"], mutant_dir, round_dir)

        if n_passing == 0:
            print(f"\n[PIPELINE] 0 sequences beat WT MPNN score in round {rnd}. Stopping early.")
            break

        # Step 3 — Relax passing mutants
        relax_mutants(mutant_dir, relaxed_dir, cfg["relax_xml"], cfg["nstruct"], cfg["n_workers"])

        # Step 4 — Collect rep1 relaxed structures for next round input
        n_collected = collect_best_relaxed(relaxed_dir, passing_dir, cfg)

        if n_collected == 0:
            print(f"\n[PIPELINE] No relaxed pdbs collected in round {rnd}. Stopping early.")
            break

        current_input_dir = passing_dir
        print(f"\n[PIPELINE] Round {rnd} complete. {n_collected} structure(s) passing to round {rnd + 1}.")

    print("\n[PIPELINE] Enrichment complete.")


if __name__ == "__main__":
    main()