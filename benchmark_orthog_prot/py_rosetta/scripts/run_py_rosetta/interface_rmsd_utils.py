"""Interface-restricted RMSD: same matched-residue RMSD as
pyrosetta_utils.global_unaligned_rmsd, but scored over interface residues only
(both chains, within `cutoff` Angstroms of the other chain, as measured on the
reference structure). Reuses pyrosetta_utils.hotspot_residues for interface
detection -- no new alignment step, so this must be called AFTER an alignment
step such as pyrosetta_utils.global_align_pdbs, exactly like global_unaligned_rmsd.
"""

import pyrosetta as pr
from pyrosetta.rosetta.core.simple_metrics.metrics import RMSDMetric
from pyrosetta.rosetta.core.io import pose_from_pose

from pyrosetta_utils import hotspot_residues


def get_interface_residue_positions(pose, chain_a="A", chain_b="B", cutoff=10.0):
    """Pose-numbered residue positions on EITHER chain that lie within `cutoff`
    Angstroms of the other chain -- i.e. the interface from both sides."""
    a_near_b = hotspot_residues(pose, binder_chain=chain_b, target_chain=chain_a, atom_distance_cutoff=cutoff)
    b_near_a = hotspot_residues(pose, binder_chain=chain_a, target_chain=chain_b, atom_distance_cutoff=cutoff)
    return set(a_near_b) | set(b_near_a)


def interface_unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id, cutoff=10.0):
    """
    Complex-wide interface RMSD between two already-aligned structures: matched
    CA/backbone residue pairs are found the same way as global_unaligned_rmsd,
    then filtered down to residues within `cutoff` Angstroms of the other chain
    on the reference structure.

    reference_chain_id / align_chain_id: exactly two comma-separated chain
    letters, paired by position (e.g. reference_chain_id="A,B",
    align_chain_id="A,B" or "C,D").
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
    if len(reference_chain_ids) != 2:
        raise ValueError("interface_unaligned_rmsd expects exactly two chains (binder + target).")

    interface_positions = get_interface_residue_positions(
        reference_pose, chain_a=reference_chain_ids[0], chain_b=reference_chain_ids[1], cutoff=cutoff
    )

    ref_conf = reference_pose.conformation()
    mob_conf = align_pose.conformation()

    reference_residue_indices = pr.rosetta.utility.vector1_unsigned_long()
    align_residue_indices = pr.rosetta.utility.vector1_unsigned_long()

    for ref_cid_letter, align_cid_letter in zip(reference_chain_ids, align_chain_ids):
        reference_chain = pr.rosetta.core.pose.get_chain_id_from_chain(ref_cid_letter, reference_pose)
        align_chain = pr.rosetta.core.pose.get_chain_id_from_chain(align_cid_letter, align_pose)

        ref_begin = ref_conf.chain_begin(reference_chain)
        ref_end = ref_conf.chain_end(reference_chain)
        mob_begin = mob_conf.chain_begin(align_chain)
        mob_end = mob_conf.chain_end(align_chain)

        len_ref = ref_end - ref_begin + 1
        len_mob = mob_end - mob_begin + 1
        k = min(len_ref, len_mob)

        # Match residues positionally within this chain (same convention as
        # global_align_pdbs / global_unaligned_rmsd), then keep only interface positions
        for off in range(k):
            r = ref_begin + off
            m = mob_begin + off
            if r not in interface_positions:
                continue
            rsd_r = reference_pose.residue(r)
            rsd_m = align_pose.residue(m)
            if rsd_r.is_protein() and rsd_m.is_protein():
                reference_residue_indices.append(r)
                align_residue_indices.append(m)

    if len(reference_residue_indices) == 0:
        raise ValueError("No interface residues found within the matched chains.")

    reference_chain_pose = pr.Pose()
    align_chain_pose = pr.Pose()
    pose_from_pose(reference_chain_pose, reference_pose, reference_residue_indices)
    pose_from_pose(align_chain_pose, align_pose, align_residue_indices)

    rmsd_metric = RMSDMetric()
    rmsd_metric.set_comparison_pose(reference_chain_pose)
    rmsd = rmsd_metric.calculate(align_chain_pose)

    return round(rmsd, 2)
