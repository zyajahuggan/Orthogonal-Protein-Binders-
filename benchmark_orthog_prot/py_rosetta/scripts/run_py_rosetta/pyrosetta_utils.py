"""
PyRosetta utility functions from BindCraft.

Attribution:
This module is adapted from the PyRosetta utilities in the BindCraft project:
https://github.com/martinpacesa/BindCraft

If you use these utilities, please cite:
Pacesa, M., Nickel, L., Schellhaas, C. et al. One-shot design of functional protein binders with BindCraft. Nature (2025). https://doi.org/10.1038/s41586-025-09429-6

"""

import os
import multiprocessing
import pyrosetta as pr
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.select import get_residues_from_subset
from pyrosetta.rosetta.core.io import pose_from_pose
from pyrosetta import pose_from_pdb
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.pose import Pose
from pyrosetta.rosetta.core.select.residue_selector import ResidueIndexSelector
from pyrosetta.rosetta.core.select.residue_selector import NeighborhoodResidueSelector
from collections import defaultdict
import numpy as np
from pyrosetta.rosetta.core.pack.guidance_scoreterms.sap import calculate_per_res_sap
from pyrosetta.rosetta.core.select.residue_selector import TrueResidueSelector
from pyrosetta.rosetta.core.select.residue_selector import (
    SecondaryStructureSelector,
    ChainSelector,
    AndResidueSelector,
)
from pyrosetta.rosetta.core.scoring.sc import ShapeComplementarityCalculator
from pyrosetta.rosetta.core.scoring.dssp import Dssp

from pyrosetta.rosetta.core.select.residue_selector import (
    SecondaryStructureSelector,
    ChainSelector,
    AndResidueSelector,
    selection_positions
)
from pathlib import Path 
import re 
import pyrosetta as pr
from Bio import Align
from Bio.Align import substitution_matrices
from scipy.optimize import linear_sum_assignment


def get_chain_sequences(pose):
    """
    Returns a dict mapping chain letter -> one-letter sequence string
    for every protein chain in the pose.
    """
    chain_seqs = {}
    conf = pose.conformation()
    n_chains = pose.num_chains()

    for chain_num in range(1, n_chains + 1):
        begin = conf.chain_begin(chain_num)
        end = conf.chain_end(chain_num)

        # Get the PDB chain letter for this Rosetta chain number
        chain_letter = pr.rosetta.core.pose.get_chain_from_chain_id(chain_num, pose)

        seq = ""
        for r in range(begin, end + 1):
            rsd = pose.residue(r)
            if rsd.is_protein():
                seq += rsd.name1()

        if seq:  # skip empty/non-protein chains (ligands, waters, etc.)
            # if a letter somehow repeats (rare), keep the longest sequence under that letter
            if chain_letter not in chain_seqs or len(seq) > len(chain_seqs[chain_letter]):
                chain_seqs[chain_letter] = seq

    return chain_seqs


def match_chains_by_sequence(reference_pose, align_pose, min_identity=0.0):
    """
    Determines the best one-to-one correspondence between chains of
    reference_pose and align_pose based on global sequence alignment
    identity (percentage of matched positions over the alignment length).

    Returns:
        reference_chain_ids (str): comma-separated chain letters, e.g. "A,B"
        align_chain_ids (str):     comma-separated chain letters, e.g. "B,A"
        score_matrix (dict):       {(ref_letter, align_letter): identity_fraction}
                                    for inspection/debugging
    """
    ref_seqs = get_chain_sequences(reference_pose)
    align_seqs = get_chain_sequences(align_pose)

    ref_letters = list(ref_seqs.keys())
    align_letters = list(align_seqs.keys())

    if not ref_letters or not align_letters:
        raise ValueError("No protein chains with sequence found in one or both poses.")

    # Set up a global aligner with BLOSUM62 for identity scoring
    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    aligner.mode = "global"

    # Build identity matrix: rows = reference chains, cols = align chains
    n_ref = len(ref_letters)
    n_align = len(align_letters)
    identity_matrix = [[0.0] * n_align for _ in range(n_ref)]
    score_matrix = {}

    for i, r_letter in enumerate(ref_letters):
        r_seq = ref_seqs[r_letter]
        for j, a_letter in enumerate(align_letters):
            a_seq = align_seqs[a_letter]

            alignment = aligner.align(r_seq, a_seq)[0]
            aligned_ref, aligned_align = str(alignment[0]), str(alignment[1])

            matches = sum(
                1 for x, y in zip(aligned_ref, aligned_align)
                if x == y and x != "-"
            )
            aln_len = len(aligned_ref)
            identity = matches / aln_len if aln_len > 0 else 0.0

            # Use negative identity since linear_sum_assignment MINIMIZES cost
            identity_matrix[i][j] = -identity
            score_matrix[(r_letter, a_letter)] = identity

    # Solve optimal one-to-one assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(identity_matrix)

    matched_ref = []
    matched_align = []
    for i, j in zip(row_ind, col_ind):
        r_letter = ref_letters[i]
        a_letter = align_letters[j]
        identity = score_matrix[(r_letter, a_letter)]
        if identity >= min_identity:
            matched_ref.append(r_letter)
            matched_align.append(a_letter)
        else:
            print(f"Warning: best match for ref chain {r_letter} is align chain "
                  f"{a_letter} but identity ({identity:.2f}) is below min_identity "
                  f"({min_identity}); excluding this pair.")

    if not matched_ref:
        raise ValueError("No chain pairs met the min_identity threshold.")

    reference_chain_ids = ",".join(matched_ref)
    align_chain_ids = ",".join(matched_align)

    return reference_chain_ids, align_chain_ids, score_matrix


def global_align_pdbs(reference_pdb, align_pdb, reference_chain_id=None, align_chain_id=None,
               auto_match=False, min_identity=0.0):
    """
    Align align_pdb onto reference_pdb using matched CA atoms from one or more
    chain pairs.

    Args:
        reference_pdb: path to reference structure PDB
        align_pdb:     path to structure that will be moved/overwritten
        reference_chain_id: comma-separated chain letters in reference_pdb, e.g. "A,B"
                            (ignored if auto_match=True)
        align_chain_id:     comma-separated chain letters in align_pdb, e.g. "A,B"
                            (ignored if auto_match=True)
        auto_match:    if True, ignore reference_chain_id/align_chain_id and instead
                       determine chain correspondence automatically via sequence identity
        min_identity:  minimum sequence identity (0-1) required to accept an
                       auto-matched chain pair (only used if auto_match=True)
    """
    # initiate poses
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # Determine chain correspondence
    if auto_match:
        reference_chain_id, align_chain_id, scores = match_chains_by_sequence(
            reference_pose, align_pose, min_identity=min_identity
        )
        print(f"Auto-matched chains: reference={reference_chain_id}, align={align_chain_id}")
        print("Identity scores:", scores)
    else:
        if reference_chain_id is None or align_chain_id is None:
            raise ValueError(
                "Must provide reference_chain_id and align_chain_id, "
                "or set auto_match=True."
            )

    reference_chain_ids = reference_chain_id.split(",")
    align_chain_ids     = align_chain_id.split(",")

    if len(reference_chain_ids) != len(align_chain_ids):
        raise ValueError(
            f"Number of reference chains ({len(reference_chain_ids)}) must match "
            f"number of align chains ({len(align_chain_ids)}) — they're paired by position."
        )

    ref_conf = reference_pose.conformation()
    mob_conf = align_pose.conformation()

    mob_ids = pr.rosetta.utility.vector1_core_id_AtomID()
    ref_ids = pr.rosetta.utility.vector1_core_id_AtomID()

    # Loop over each paired chain (e.g. A<->A, B<->B, or A<->C, B<->D, etc.)
    for ref_cid_letter, align_cid_letter in zip(reference_chain_ids, align_chain_ids):
        reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(ref_cid_letter, reference_pose)
        align_chain     = pr.rosetta.core.pose.get_chain_id_from_chain(align_cid_letter, align_pose)

        ref_begin = ref_conf.chain_begin(reference_chain)
        ref_end   = ref_conf.chain_end(reference_chain)
        mob_begin = mob_conf.chain_begin(align_chain)
        mob_end   = mob_conf.chain_end(align_chain)

        len_ref = ref_end - ref_begin + 1
        len_mob = mob_end - mob_begin + 1
        k = min(len_ref, len_mob)

        for off in range(k):
            r = ref_begin + off
            m = mob_begin + off
            rsd_r = reference_pose.residue(r)
            rsd_m = align_pose.residue(m)
            if rsd_r.is_protein() and rsd_r.has("CA") and rsd_m.is_protein() and rsd_m.has("CA"):
                ref_ids.append(pr.rosetta.core.id.AtomID(rsd_r.atom_index("CA"), r))
                mob_ids.append(pr.rosetta.core.id.AtomID(rsd_m.atom_index("CA"), m))

    # Need at least 3 pairs; use len(...) for vector1_* containers
    if len(mob_ids) < 3:
        raise ValueError("Not enough matched CA pairs to superimpose (need ≥3).")
    if len(mob_ids) != len(ref_ids):
        raise ValueError("Internal error: mobile/ref CA pair counts differ.")

    # ---- superimpose using ALL matched pairs across ALL chains ----
    atom_map = pr.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    for a_m, a_r in zip(mob_ids, ref_ids):   # MOBILE -> REFERENCE
        atom_map[a_m] = a_r

    pr.rosetta.core.scoring.superimpose_pose(
        align_pose,
        reference_pose,
        atom_map
    )
    # -----------------------------------------------------------------

    # Save aligned PDB
    align_pose.dump_pdb(align_pdb)
    clean_pdb(align_pdb)

def global_unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Computes complex-wide Cα/backbone RMSD (via RMSDMetric) between one or more
    matched chain pairs, without performing any alignment first.

    reference_chain_id / align_chain_id: comma-separated chain letters, paired by
    position (e.g. reference_chain_id="A,B", align_chain_id="A,B" or "C,D").
    """
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    reference_chain_ids = reference_chain_id.split(",")
    align_chain_ids = align_chain_id.split(",")

    if len(reference_chain_ids) != len(align_chain_ids):
        raise ValueError(
            f"Number of reference chains ({len(reference_chain_ids)}) must match "
            f"number of align chains ({len(align_chain_ids)}) — they're paired by position."
        )

    ref_conf = reference_pose.conformation()
    mob_conf = align_pose.conformation()

    # Accumulate matched residue indices across ALL chain pairs
    reference_residue_indices = pr.rosetta.utility.vector1_unsigned_long()
    align_residue_indices = pr.rosetta.utility.vector1_unsigned_long()

    for ref_cid_letter, align_cid_letter in zip(reference_chain_ids, align_chain_ids):
        try:
            reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(ref_cid_letter, reference_pose)
        except RuntimeError:
            raise ValueError(f"Reference chain '{ref_cid_letter}' not found in {reference_pdb}")

        try:
            align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(align_cid_letter, align_pose)
        except RuntimeError:
            # Fallback: use the last chain in align_pose if the requested letter isn't found
            n_chains = align_pose.num_chains()
            last_res = align_pose.conformation().chain_end(n_chains)
            fallback_letter = align_pose.pdb_info().chain(last_res)
            print(f"Warning: align chain '{align_cid_letter}' not found in {align_pdb}; "
                  f"falling back to chain '{fallback_letter}'")
            align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(fallback_letter, align_pose)

        ref_begin = ref_conf.chain_begin(reference_chain)
        ref_end   = ref_conf.chain_end(reference_chain)
        mob_begin = mob_conf.chain_begin(align_chain)
        mob_end   = mob_conf.chain_end(align_chain)

        len_ref = ref_end - ref_begin + 1
        len_mob = mob_end - mob_begin + 1
        k = min(len_ref, len_mob)

        # Match residues positionally within this chain (same convention as align_pdbs)
        for off in range(k):
            r = ref_begin + off
            m = mob_begin + off
            rsd_r = reference_pose.residue(r)
            rsd_m = align_pose.residue(m)
            if rsd_r.is_protein() and rsd_m.is_protein():
                reference_residue_indices.append(r)
                align_residue_indices.append(m)

    if len(reference_residue_indices) == 0:
        raise ValueError("No matched protein residues found across the specified chains.")
    if len(reference_residue_indices) != len(align_residue_indices):
        raise ValueError("Internal error: reference/align residue index counts differ.")

    # Build combined multi-chain subposes
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # Calculate complex-wide RMSD using the RMSDMetric
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)


def get_cognate_status_cross_docking(design_name):
    ligand, receptor = design_name.split("_vs_")
    cognate_status = 0 
    if ligand + "s" == receptor:
            cognate_status = 1
    return cognate_status

    
def get_cognate_status_dhd(name):
    cognate_status = 0
    pattern = re.compile(r"^(.+)(?:a_vs_\1b|b_vs_\1a)$")
    found = pattern.search(name)
    if found is not None:
        cognate_status = 1
    return cognate_status

def get_protein_name(pdb_path):
    path = Path(pdb_path).resolve()
    if "af3" in str(path):
        protein_pair = path.stem[:-6]
    elif "chai" in str(path):
        protein_pair = path.parent.name
    elif "esmfold2" in str(path):
        protein_pair = path.stem.replace("__", "_vs_")
    elif "boltz" in str(path):
        protein_pair = path.stem[:-8]
    return protein_pair 


def clean_pdb(pdb_file):
    """Remove unnecessary information from PDB files for clean processing.
    
    Filters PDB files to retain only essential structural information by
    keeping ATOM, HETATM, MODEL, TER, and END records while removing
    Rosetta-specific annotations and other metadata.
    
    Args:
        pdb_file (str): Path to the PDB file to be cleaned
    """
    # Read the pdb file and filter relevant lines
    with open(pdb_file, "r") as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def hotspot_residues(pose:Pose, binder_chain:str, target_chain:str, atom_distance_cutoff=10.0):
    b_sel   = ChainSelector(binder_chain)
    t_sel   = ChainSelector(target_chain)
    t_mask  = AndResidueSelector(NeighborhoodResidueSelector(b_sel, atom_distance_cutoff, False), t_sel).apply(pose)

    positions = selection_positions(t_mask)
    return {pos: pose.residue(pos).name1() for pos in positions}

def calculate_loop_sc(pose, binder_chain="B", target_chain="A"):
    dssp = Dssp(pose)
    dssp.insert_ss_into_pose(pose)

    ss_selector = SecondaryStructureSelector()
    ss_selector.set_selected_ss("L")
    chain_selector = ChainSelector(binder_chain)
    loop_selector = AndResidueSelector(ss_selector, chain_selector)
    residue_mask = loop_selector.apply(pose)
    loop_residues = [i for i, selected in enumerate(residue_mask, 1) if selected]

    # No loop residues in binder -> SC is undefined, bail out gracefully
    if not loop_residues:
        return None, None

    sc_calc = ShapeComplementarityCalculator()
    sc_calc.Init()

    for res_id in loop_residues:
        sc_calc.AddResidue(1, pose.residue(res_id))

    target_chain = target_chain.split(",") if isinstance(target_chain, str) else target_chain
    added_mol2 = 0
    for res_id in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(res_id) in target_chain:  # use pdb chain LETTER, not pose.chain() int
            sc_calc.AddResidue(2, pose.residue(res_id))
            added_mol2 += 1

    if added_mol2 == 0:
        return None, None

    sc_calc.Calc()
    results = sc_calc.GetResults()
    return results.sc, results.area


# Rosetta interface scores
def score_interface(pdb_file, binder_chain="B", target_chain="A"):
    # load pose
    pose = pr.pose_from_pdb(pdb_file)

    # analyze interface statistics
    iam = InterfaceAnalyzerMover()
    scorefxn = pr.get_fa_scorefxn()
    docking_partners = pr.rosetta.core.pose.DockingPartners.docking_partners_from_string(f"{target_chain}_{binder_chain}")
    iam.set_interface(docking_partners)
    iam.set_scorefunction(scorefxn)
    iam.set_compute_packstat(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_separated(True)
    iam.apply(pose)

    # Initialize dictionary with all amino acids
    interface_AA = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}

    # Initialize list to store PDB residue IDs at the interface
    interface_residues_set = hotspot_residues(
            pose, binder_chain, target_chain=target_chain
    )
    interface_residues_pdb_ids = []

    # Iterate over the interface residues
    for pdb_res_num, aa_type in interface_residues_set.items():
        # Increase the count for this amino acid type
        interface_AA[aa_type] += 1

        # Append the binder_chain and the PDB residue number to the list
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # count interface residues
    interface_nres = len(interface_residues_pdb_ids)

    # Convert the list into a comma-separated string
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)

    # Calculate the percentage of hydrophobic residues at the interface of the binder
    hydrophobic_aa = set("ACFILMPVWY")
    hydrophobic_count = sum(interface_AA[aa] for aa in hydrophobic_aa)
    if interface_nres != 0:
        interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100
    else:
        interface_hydrophobicity = 0

    # retrieve statistics
    interfacescore = iam.get_all_data()
    interface_sc = interfacescore.sc_value  # shape complementarity
    interface_loop_sc, interface_loop_sc_area = calculate_loop_sc(
        pose, binder_chain, target_chain
    )
    interface_interface_hbonds = (
        interfacescore.interface_hbonds
    )  # number of interface H-bonds
    interface_dG = iam.get_interface_dG()  # interface dG
    interface_dSASA = (
        iam.get_interface_delta_sasa()
    )  # interface dSASA (interface surface area)
    interface_packstat = iam.get_interface_packstat()  # interface pack stat score
    interface_dG_SASA_ratio = (
        interfacescore.dG_dSASA_ratio * 100
    )  # ratio of dG/dSASA (normalised energy for interface area size)
    buns_filter = XmlObjects.static_get_filter(
        '<BuriedUnsatHbonds report_all_heavy_atom_unsats="true" scorefxn="scorefxn" ignore_surface_res="false" use_ddG_style="true"  probe_radius="1.1" burial_cutoff_apo="0.2" confidence="0" />'
    )
    interface_delta_unsat_hbonds = buns_filter.report_sm(pose)

    if interface_nres != 0:
        interface_hbond_percentage = (
            interface_interface_hbonds / interface_nres
        ) * 100  # Hbonds per interface size percentage
        interface_bunsch_percentage = (
            interface_delta_unsat_hbonds / interface_nres
        ) * 100  # Unsaturated H-bonds per percentage
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # calculate binder energy score
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn)
    tem.set_residue_selector(chain_design)
    binder_score = tem.calculate(pose)

    # calculate binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = bsasa.calculate(pose)

    if binder_sasa > 0:
        interface_binder_fraction = (interface_dSASA / binder_sasa) * 100
    else:
        interface_binder_fraction = 0

    # calculate surface hydrophobicity
    binder_pose = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }[binder_chain]

    layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
    layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
    surface_res = layer_sel.apply(binder_pose)

    exp_apol_count = 0
    total_count = 0

    # count apolar and aromatic residues at the surface
    for i in range(1, len(surface_res) + 1):
        if surface_res[i] == True:
            res = binder_pose.residue(i)

            # count apolar and aromatic residues as hydrophobic
            if (
                res.is_apolar() == True
                or res.name() == "PHE"
                or res.name() == "TRP"
                or res.name() == "TYR"
            ):
                exp_apol_count += 1
            total_count += 1

    surface_hydrophobicity = exp_apol_count / total_count

    # output interface score array and amino acid counts at the interface
    interface_scores = {
        "binder_score": binder_score,
        "surface_hydrophobicity": surface_hydrophobicity,
        "interface_sc": interface_sc,
        "interface_loop_sc": interface_loop_sc,
        "interface_loop_sc_area": interface_loop_sc_area,
        "interface_packstat": interface_packstat,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio,
        "interface_fraction": interface_binder_fraction,
        "interface_hydrophobicity": interface_hydrophobicity,
        "interface_nres": interface_nres,
        "interface_interface_hbonds": interface_interface_hbonds,
        "interface_hbond_percentage": interface_hbond_percentage,
        "interface_delta_unsat_hbonds": interface_delta_unsat_hbonds,
        "interface_delta_unsat_hbonds_percentage": interface_bunsch_percentage,
    }

    # round to two decimal places
    interface_scores = {
        k: round(v, 2) if isinstance(v, float) else v
        for k, v in interface_scores.items()
    }

    return interface_scores #, interface_AA, interface_residues_pdb_ids_str 


# align pdbs to have same orientation
def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    # initiate poses
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # Take the first token if "A,B" is passed
    reference_chain_id = reference_chain_id.split(",")[0]
    align_chain_id     = align_chain_id.split(",")[0]

    # Chain numbers in these poses
    reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(reference_chain_id, reference_pose)
    align_chain     = pr.rosetta.core.pose.get_chain_id_from_chain(align_chain_id, align_pose)

    # Chain ranges
    ref_conf = reference_pose.conformation()
    mob_conf = align_pose.conformation()
    ref_begin = ref_conf.chain_begin(reference_chain)
    ref_end   = ref_conf.chain_end(reference_chain)
    mob_begin = mob_conf.chain_begin(align_chain)
    mob_end   = mob_conf.chain_end(align_chain)

    # Equal-length contiguous segment from starts of each chain
    len_ref = ref_end - ref_begin + 1
    len_mob = mob_end - mob_begin + 1
    k = min(len_ref, len_mob)

    # Build matching CA pairs (only add when both residues are protein and have CA)
    mob_ids = pr.rosetta.utility.vector1_core_id_AtomID()
    ref_ids = pr.rosetta.utility.vector1_core_id_AtomID()
    for off in range(k):
        r = ref_begin + off
        m = mob_begin + off 
        rsd_r = reference_pose.residue(r)
        rsd_m = align_pose.residue(m)
        if rsd_r.is_protein() and rsd_r.has("CA") and rsd_m.is_protein() and rsd_m.has("CA"):
            ref_ids.append(pr.rosetta.core.id.AtomID(rsd_r.atom_index("CA"), r))
            mob_ids.append(pr.rosetta.core.id.AtomID(rsd_m.atom_index("CA"), m))

    # Need at least 3 pairs; use len(...) for vector1_* containers
    if len(mob_ids) < 3:
        raise ValueError("Not enough matched CA pairs to superimpose (need ≥3).")
    if len(mob_ids) != len(ref_ids):
        raise ValueError("Internal error: mobile/ref CA pair counts differ.")

    # ---- superimpose using the std::map overload (works on your build) ----
    atom_map = pr.rosetta.std.map_core_id_AtomID_core_id_AtomID()
    for a_m, a_r in zip(mob_ids, ref_ids):   # MOBILE -> REFERENCE
        atom_map[a_m] = a_r

    pr.rosetta.core.scoring.superimpose_pose(
        align_pose,
        reference_pose,
        atom_map
    )
    # ----------------------------------------------------------------------

    # Save aligned PDB
    align_pose.dump_pdb(align_pdb)
    clean_pdb(align_pdb)


def get_sap_score(
    pdb,
    binder_chain=None,
    only_binder=False,
    hydrophobic_aa=None,
    patch_radius=8,
    limit_sasa=1,
    avg_sasa_patch_thr=0.75,
    cdrs=None,
):
    pose_ = pr.pose_from_pdb(
        pdb
    )  # Assuming 'pdb' is defined and contains your PDB file path

    selector = TrueResidueSelector()
    if binder_chain is not None and only_binder:
        idxs = {"A": 1, "B": 2, "C": 3, "D": 4}
        pose = Pose()
        pose = pose_.split_by_chain(idxs[binder_chain])
    else:
        pose = pose_
        if binder_chain is not None:
            selector = ChainSelector(binder_chain)

    # Create vectors to store the results
    num_residues = pose.total_residue()

    if hydrophobic_aa is None:
        # Top from hydrophobic scale (Black S.D., Mould D.R.)
        hydrophobic_aa = ["LEU", "ILE", "PHE", "TRP", "VAL", "MET", "TYR", "ALA"]

    scale = {
        "ALA": 0.37,
        "ARG": -1.52,
        "ASN": -0.79,
        "ASP": -1.43,
        "CYS": 0.55,
        "GLN": -0.76,
        "GLU": -1.40,
        "GLY": 0.00,
        "HIS": -1.00,
        "ILE": 1.34,
        "LEU": 1.34,
        "LYS": -0.67,
        "MET": 0.73,
        "PHE": 1.52,
        "PRO": 0.64,
        "SER": -0.43,
        "THR": -0.15,
        "TRP": 1.16,
        "TYR": 1.16,
        "VAL": 1.00,
    }

    sap_score = calculate_per_res_sap(
        pose=pose, score_sel=selector, sap_calculate_sel=selector, sasa_sel=selector
    )

    def avg_sap_hydrophobic_patch(sap_score, residues):
        avg_sap = 0
        for r in residues:
            avg_sap += sap_score[r[0]]  # * scale[r[1]]
        avg_sap = avg_sap / len(residues)
        return avg_sap

    def patch_exists(hydrophobic_patches, nearby_res):
        if len(hydrophobic_patches) < 1:
            return False
        else:
            nrb = set(nearby_res)
            for hp in hydrophobic_patches:
                prev = set(hp[1])
                if len(nrb - prev) <= len(nrb) - 2:
                    return True
            return False

    exposed_hydrophobic_aa = []
    hydrophobic_patches = []
    for i in range(1, num_residues + 1):
        aa_type = pose.residue(i).name3()  # Get three letter code
        if binder_chain is not None and pose.pdb_info().chain(i) != binder_chain:
            continue
        if aa_type in hydrophobic_aa and (sap_score[i]) >= limit_sasa:
            exposed_hydrophobic_aa.append((i, aa_type))
            nearby_res = get_nearby_residues(pose, i, distance=patch_radius)
            avg_sap_patch = avg_sap_hydrophobic_patch(sap_score, nearby_res)

            if avg_sap_patch >= avg_sasa_patch_thr:
                if not patch_exists(hydrophobic_patches, nearby_res):
                    hydrophobic_patches.append((avg_sap_patch, nearby_res))

    cdr_sap = np.array(sap_score)
    if not cdrs is None:
        cdr_sap = sum(cdr_sap[cdrs])
    else:
        cdr_sap = sum(cdr_sap)

    return sum(sap_score), cdr_sap, exposed_hydrophobic_aa, hydrophobic_patches


def get_nearby_residues(pose, target_residue_number, distance=8.0):
    """
    Get all residues within a specified distance of a target residue

    Args:
        pose: PyRosetta Pose object
        target_residue_number: int, the residue number you're interested in
        distance: float, the cutoff distance in Angstroms (default 8.0Å)

    Returns:
        list of residue numbers that are within the specified distance
    """
    # Create selector for the target residue
    target_selector = ResidueIndexSelector(target_residue_number)

    # Create neighborhood selector
    neighbor_selector = NeighborhoodResidueSelector()
    neighbor_selector.set_focus_selector(target_selector)
    neighbor_selector.set_distance(distance)

    # Apply selector to get nearby residues
    nearby_residues = []
    for i in range(1, pose.total_residue() + 1):
        if neighbor_selector.apply(pose)[i]:
            nearby_residues.append((i, pose.residue(i).name3()))

    return nearby_residues


# calculate the rmsd without alignment
def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    reference_pose = pr.pose_from_pdb(reference_pdb)
    align_pose = pr.pose_from_pdb(align_pdb)

    # Define chain selectors for the reference and align chains
    reference_chain_selector = ChainSelector(reference_chain_id)
    align_chain_selector = ChainSelector(align_chain_id)

    # Apply selectors to get residue subsets
    reference_chain_subset = reference_chain_selector.apply(reference_pose)
    align_chain_subset = align_chain_selector.apply(align_pose)

    if not any(align_chain_subset):
        n_chains = align_pose.num_chains()
        last_res = align_pose.conformation().chain_end(n_chains)
        fallback_chain_id = align_pose.pdb_info().chain(last_res)
        align_chain_selector = ChainSelector(fallback_chain_id)
        align_chain_subset = align_chain_selector.apply(align_pose)

    # Convert subsets to residue index vectors
    reference_residue_indices = get_residues_from_subset(reference_chain_subset)
    align_residue_indices = get_residues_from_subset(align_chain_subset)

    # Create empty subposes
    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()

    # Fill subposes
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    # Calculate RMSD using the RMSDMetric
    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)


def _relax_worker(pdb_file, relaxed_pdb_path, seed):
    """Worker function for parallel Rosetta relax. Runs in a child process
    with its own PyRosetta initialization.

    On exception, writes a full traceback to ``{relaxed_pdb_path}.err`` so the
    parent (``pr_relax_parallel``) can read and print it. Without this, child
    failures only surface as a missing output file with no diagnostics.
    """
    import traceback
    try:
        pr.init(
            f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
            f"-corrections::beta_nov16 true -relax:default_repeats 1 "
            f"-run:constant_seed -run:jran {seed}"
        )

        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        mmf = MoveMap()
        mmf.set_chi(True)
        mmf.set_bb(True)
        mmf.set_jump(False)

        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf)
        fastrelax.max_iter(200)
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)
    except Exception:
        err_path = f"{relaxed_pdb_path}.err"
        try:
            with open(err_path, "w") as fh:
                fh.write(
                    f"_relax_worker failed for pdb_file={pdb_file} seed={seed}\n\n"
                )
                fh.write(traceback.format_exc())
        except Exception:
            pass
        raise


def pr_relax_parallel(pdb_file, output_dir, design_name,  n_relax=5):
    """Run parallel Rosetta FastRelax with different random seeds.

    Spawns n_relax child processes, each with its own PyRosetta initialization
    and a unique random seed. Note: each spawn re-initializes PyRosetta (~30-60s
    overhead per process), so n_relax should be kept small (3-5).

    Returns:
        list[str]: Paths to relaxed PDB files (only those that succeeded).
    """
    ctx = multiprocessing.get_context("spawn")
    relaxed_paths = []
    processes = []
    seeds = np.random.randint(0, 999999, size=n_relax).tolist()

    for i, seed in enumerate(seeds):
        relaxed_pdb_path = os.path.join(output_dir, f"{design_name}_relaxed_{i}.pdb")
        relaxed_paths.append(relaxed_pdb_path)

        if os.path.exists(relaxed_pdb_path):
            continue

        p = ctx.Process(
            target=_relax_worker,
            args=(pdb_file, relaxed_pdb_path, seed),
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    missing = [p for p in relaxed_paths if not os.path.exists(p)]
    if missing:
        print(
            f"\n{'=' * 78}\n"
            f"[RELAX ERROR] {len(missing)}/{len(relaxed_paths)} parallel relax runs FAILED\n"
            f"{'=' * 78}",
            flush=True,
        )
        for p in missing:
            err_path = f"{p}.err"
            print(f"\n--- FAILED: {p} ---", flush=True)
            if os.path.exists(err_path):
                try:
                    with open(err_path) as fh:
                        print(fh.read(), flush=True)
                except Exception as exc:
                    print(f"(could not read {err_path}: {exc})", flush=True)
                # Clean up: traceback already in this log; no need to keep
                # the .err file across runs (they accumulate otherwise).
                try:
                    os.remove(err_path)
                except OSError:
                    pass
            else:
                print(
                    f"(no traceback file at {err_path} — child likely died "
                    f"before reaching the except block)",
                    flush=True,
                )
        print(f"{'=' * 78}\n", flush=True)
        relaxed_paths = [p for p in relaxed_paths if os.path.exists(p)]

    return relaxed_paths


def score_interface_ensemble(
    relaxed_pdb_paths, binder_chain="B", target_chain="A", score_mode="average"
):
    """Score interface metrics across an ensemble of relaxed structures.

    Selects best structure by lowest binder_score (most negative Rosetta energy).

    Args:
        score_mode: "average" — average numeric metrics across all runs;
                    "best" — return metrics from the single best structure only.

    Returns:
        tuple: (interface_scores, best_interface_AA, best_interface_residues, best_relaxed_pdb_path)
    """
    #all_scores, all_aa, all_residues = [], [], []
    all_scores = []

    for pdb_path in relaxed_pdb_paths:
        try:
            #scores, aa, residues = score_interface(pdb_path, binder_chain, target_chain)
            scores = score_interface(pdb_path, binder_chain, target_chain)
            all_scores.append(scores)
            #all_aa.append(aa)
            #all_residues.append(residues)
        except Exception as e:
            print(f"Warning: score_interface failed for {pdb_path}: {e}")

    if not all_scores:
        raise RuntimeError("All score_interface calls failed in ensemble scoring")

    best_idx = np.argmin([s["binder_score"] for s in all_scores]) #finding lowest binder score index
    #best_interface_AA = all_aa[best_idx]
    #best_interface_residues = all_residues[best_idx]
    best_relaxed_pdb_path = relaxed_pdb_paths[best_idx]

    if score_mode == "best":
        return best_relaxed_pdb_path #,result_scores , best_interface_AA, best_interface_residues

    result_scores = {}
    for key in all_scores[0]:
        values = [s[key] for s in all_scores if s.get(key) is not None]
        if values and isinstance(values[0], (int, float)):
            result_scores[key] = np.mean(values)
        else:
            result_scores[key] = all_scores[0][key]

    return best_relaxed_pdb_path #,result_scores , best_interface_AA, best_interface_residues


# Relax designed structure
def pr_relax(pdb_file, relaxed_pdb_path):
    if not os.path.exists(relaxed_pdb_path):
        # Generate pose
        pose = pr.pose_from_pdb(pdb_file)
        start_pose = pose.clone()

        ### Generate movemaps
        mmf = MoveMap()
        mmf.set_chi(True)  # enable sidechain movement
        mmf.set_bb(
            True
        )  # enable backbone movement, can be disabled to increase speed by 30% but makes metrics look worse on average
        mmf.set_jump(False)  # disable whole chain movement

        # Run FastRelax
        fastrelax = FastRelax()
        scorefxn = pr.get_fa_scorefxn()
        fastrelax.set_scorefxn(scorefxn)
        fastrelax.set_movemap(mmf)  # set MoveMap
        fastrelax.max_iter(200)  # default iterations is 2500
        fastrelax.min_type("lbfgs_armijo_nonmonotone")
        fastrelax.constrain_relax_to_start_coords(True)
        fastrelax.apply(pose)

        # Align relaxed structure to original trajectory
        align = AlignChainMover()
        align.source_chain(0)
        align.target_chain(0)
        align.pose(start_pose)
        align.apply(pose)

        # Copy B factors from start_pose to pose
        for resid in range(1, pose.total_residue() + 1):
            if pose.residue(resid).is_protein():
                # Get the B factor of the first heavy atom in the residue
                bfactor = start_pose.pdb_info().bfactor(resid, 1)
                for atom_id in range(1, pose.residue(resid).natoms() + 1):
                    pose.pdb_info().bfactor(resid, atom_id, bfactor)

        # output relaxed and aligned PDB
        pose.dump_pdb(relaxed_pdb_path)
        clean_pdb(relaxed_pdb_path)


def get_chain_length(pose, chain_id="A"):
    """
    Get the number of residues in a specific chain

    Parameters:
    - pose: PyRosetta pose
    - chain_id: Chain identifier (default 'A')

    Returns:
    - Number of residues in the chain
    """
    chain_length = 0
    for i in range(1, pose.total_residue() + 1):
        if pose.pdb_info().chain(i) == chain_id:
            chain_length += 1
    return chain_length


def get_cb_coordinates(residue):
    """Get CB coordinates (CA for GLY)"""
    if residue.name3() == "GLY":
        return np.array(residue.xyz("CA"))
    return np.array(residue.xyz("CB"))


def get_key_atoms(residue_name):
    """
    Get key atoms to check for each residue type
    Returns list of important atoms for VDW contacts
    """
    # Backbone atoms for all residues
    backbone = ["CA"]  # Only using CA and CB from backbone for efficiency

    # Side chain atoms by residue type
    side_chain = {
        # Hydrophobic residues
        "VAL": ["CG1", "CG2"],
        "ILE": ["CG1", "CG2", "CD1"],
        "LEU": ["CG", "CD1", "CD2"],
        "MET": ["CG", "SD", "CE"],
        "ALA": [],  # CB already in backbone
        "PRO": ["CG", "CD"],
        # Aromatic residues
        "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
        "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
        "TRP": ["CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
        # Charged residues
        "ASP": ["CG", "OD1", "OD2"],
        "GLU": ["CG", "CD", "OE1", "OE2"],
        "LYS": ["CG", "CD", "CE", "NZ"],
        "ARG": ["CG", "CD", "NE", "CZ", "NH1", "NH2"],
        "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
        # Polar residues
        "SER": ["OG"],
        "THR": ["OG1", "CG2"],
        "ASN": ["CG", "OD1", "ND2"],
        "GLN": ["CG", "CD", "OE1", "NE2"],
        # Special cases
        "GLY": [],  # No side chain
        "CYS": ["SG"],
    }

    return (
        backbone
        + side_chain.get(residue_name, [])
        + (["CB"] if residue_name != "GLY" else [])
    )


def get_residue_contacts(pdb_path, chain1="A", chain2="B", cutoff_distance=4.0):
    """
    Identify all residue pairs that are in contact across the interface

    Parameters:
    - pose: PyRosetta pose
    - chain1, chain2: Chain identifiers
    - cutoff_distance: Distance cutoff for contacts (Angstroms)

    Returns:
    - Dictionary of contact information
    """
    contacts = defaultdict(list)

    pose = pose_from_pdb(pdb_path)
    target_len = pose.total_residue() - get_chain_length(pose, chain2)

    # Get residues from each chain
    chain1_residues = []
    chain2_residues = []

    # Pre-calculate CB coordinates for quick distance screening
    cb_coords = {}

    for i in range(1, pose.total_residue() + 1):
        chain = pose.pdb_info().chain(i)
        if chain == chain1:
            chain1_residues.append(i)
            cb_coords[i] = get_cb_coordinates(pose.residue(i))
        elif chain == chain2:
            chain2_residues.append(i)
            cb_coords[i] = get_cb_coordinates(pose.residue(i))

    chain1_offset = min(chain1_residues) - 1
    # Create HBond set for hydrogen bond detection
    hbond_set = pose.get_hbonds()

    # Pre-calculate all hydrogen bonds
    hbonds = defaultdict(list)
    for hbond in hbond_set.hbonds():
        don_res = hbond.don_res()
        acc_res = hbond.acc_res()
        if pose.pdb_info().chain(don_res) != pose.pdb_info().chain(acc_res):
            hbonds[(don_res, acc_res)].append(hbond)

    # Extended cutoff for initial screening (to catch all possible interactions)
    extended_cutoff = max(cutoff_distance, 5.0) + 4.0  # Add buffer for side chains

    # Analyze each residue pair, but only if CB atoms are within extended cutoff
    for res1 in chain1_residues:
        res1_obj = pose.residue(res1)
        res1_name = res1_obj.name3()

        for res2 in chain2_residues:
            # Quick CB distance check first
            if np.linalg.norm(cb_coords[res1] - cb_coords[res2]) > extended_cutoff:
                continue

            res2_obj = pose.residue(res2)
            res2_name = res2_obj.name3()

            # Initialize contact info
            contact_types = set()
            min_distance = float("inf")

            # Check representative atoms for contact instead of all atoms
            # For each residue, we'll check backbone and one or two side chain atoms
            # Get key atoms for both residues
            atoms_to_check1 = get_key_atoms(res1_name)
            atoms_to_check2 = get_key_atoms(res2_name)

            # Check distances between representative atoms
            for atom1 in atoms_to_check1:
                for atom2 in atoms_to_check2:
                    try:
                        distance = np.linalg.norm(
                            np.array(res1_obj.xyz(atom1))
                            - np.array(res2_obj.xyz(atom2))
                        )
                        if distance <= cutoff_distance:
                            min_distance = min(min_distance, distance)
                            contact_types.add("VDW Contact")
                    except KeyError:
                        continue

            # Check for hydrogen bonds (using pre-calculated hbonds)
            if (res1, res2) in hbonds or (res2, res1) in hbonds:
                contact_types.add("H-bond")
                for hbond in hbonds.get((res1, res2), []) + hbonds.get(
                    (res2, res1), []
                ):
                    don_res = pose.residue(hbond.don_res())
                    acc_res = pose.residue(hbond.acc_res())
                    don_coords = np.array(don_res.xyz(hbond.don_hatm()))
                    acc_coords = np.array(acc_res.xyz(hbond.acc_atm()))
                    hbond_distance = np.linalg.norm(don_coords - acc_coords)
                    min_distance = min(min_distance, hbond_distance)

            # Check for potential salt bridge
            if min_distance <= 4.0:  # Typical salt bridge distance
                if (
                    res1_name in ["ARG", "LYS", "HIS"] and res2_name in ["ASP", "GLU"]
                ) or (
                    res2_name in ["ARG", "LYS", "HIS"] and res1_name in ["ASP", "GLU"]
                ):
                    contact_types.add("Salt-bridge")

            # Check for potential hydrophobic interaction
            hydrophobic = [
                "ALA",
                "VAL",
                "LEU",
                "ILE",
                "MET",
                "PHE",
                "TRP",
                "PRO",
                "TYR",
            ]
            if min_distance <= 5.0:  # Typical hydrophobic interaction distance
                if res1_name in hydrophobic and res2_name in hydrophobic:
                    contact_types.add("Hydrophobic")

            # If any contacts were found, store the information
            if contact_types:
                key = (res1 - chain1_offset, res2 - target_len)
                contacts[key] = {
                    "distance": min_distance,
                    "types": sorted(list(contact_types)),
                }

    return contacts


def find_nearby_residues_from_pdb(
    pdb_path: str,
    target_residues: list[int],
    distance_threshold: float = 6.0,
    chain: str = "A",
) -> list[int]:
    """
    Find residues near target residue(s) within a specified distance threshold
    for a specific chain.

    Parameters:
    -----------
    pdb_path : str
        Path to the PDB file to be loaded
    target_residues : int or List[int]
        Sequence position(s) within the specified chain (1-indexed for that chain)
    distance_threshold : float, optional
        Maximum distance (in Angstroms) to consider a residue as "nearby"
        Default is 3.0 Angstroms
    chain : str, optional
        Chain identifier to analyze. Default is 'A'

    Returns:
    --------
    List of tuples containing:
    - Original target residue number (in chain)
    - Nearby residue sequence position (in chain)
    - Distance from the target residue
    """

    # Load the pose
    try:
        pose = pose_from_pdb(pdb_path)
    except Exception as e:
        raise ValueError(f"Error loading PDB file {pdb_path}: {e}")

    # Normalize input to a list
    if isinstance(target_residues, int):
        target_residues = [target_residues]

    # Get chain mapping
    chain_residues = {}
    current_chain = ""
    current_chain_start = 1

    for i in range(1, pose.total_residue() + 1):
        res_chain = pose.pdb_info().chain(i)

        if res_chain != current_chain:
            if current_chain:
                chain_residues[current_chain] = (current_chain_start, i - 1)
            current_chain = res_chain
            current_chain_start = i

    # Add the last chain
    chain_residues[current_chain] = (current_chain_start, pose.total_residue())

    # Validate chain exists
    if chain not in chain_residues:
        raise ValueError(
            f"Chain {chain} not found in the PDB file. Available chains: {list(chain_residues.keys())}"
        )

    # Get start and end residues for the specified chain
    chain_start, chain_end = chain_residues[chain]

    # Validate input residue numbers for the specific chain
    for res in target_residues:
        if res < 1 or res > (chain_end - chain_start + 1):
            raise ValueError(
                f"Invalid residue number {res} for chain {chain}. Chain has {chain_end - chain_start + 1} residues."
            )

    # List to store nearby residues
    nearby_residues = []

    # Iterate through each target residue
    for target_relative in target_residues:
        # Convert relative chain position to absolute pose position
        target_residue = chain_start + target_relative - 1

        # Get the 3D coordinates of the target residue's CA atom
        target_coords = pose.residue(target_residue).atom("CA").xyz()
        name_target = pose.residue(target_residue).name3()

        # Iterate through residues in the specified chain
        for res_num in range(chain_start, chain_end + 1):
            # Skip the target residue itself
            if res_num == target_residue:
                nearby_residues.append(res_num - chain_start + 1)
                continue

            # Get CA atom coordinates for the current residue
            current_coords = pose.residue(res_num).atom("CA").xyz()
            name_current = pose.residue(res_num).name3()

            # Calculate Euclidean distance
            distance = np.linalg.norm(current_coords - target_coords)

            # Check if within distance threshold
            if distance <= distance_threshold:
                # Convert absolute residue number back to chain-relative
                nearby_residues.append(res_num - chain_start + 1)

    return np.array(nearby_residues)
