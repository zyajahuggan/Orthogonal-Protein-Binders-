from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue
import numpy as np

in_pdb  = "/scratch/jgray21/zyhuggan/tryingggg/pos_neg_design/OrthoMPNN/3mjg.clean.pdb"
out_pdb = "triple_crystal_complex.pdb"

# Canonical order of chains in INPUT
input_chain_order = ["A", "B", "X", "Y"]

# Chain maps for each duplicate (ONLY 2 COPIES NOW)
maps = [
    {"A":"A", "B":"B", "X":"C", "Y":"D"},
    {"A":"E", "B":"F", "X":"G", "Y":"H"},
]

# Translation offsets so copies don't overlap
offsets = [
    np.array([0.0,   0.0, 0.0]),
    np.array([200.0, 0.0, 0.0]),
]

parser = PDBParser(QUIET=True)
orig = parser.get_structure("orig", in_pdb)
orig_model = next(orig.get_models())

# Lookup original chains
orig_chains = {ch.id: ch for ch in orig_model.get_chains()}

# Safety check
for ch in input_chain_order:
    if ch not in orig_chains:
        raise ValueError(f"Expected chain {ch} not found in input PDB")

new_struct = Structure.Structure("triple")
new_model = Model.Model(0)
new_struct.add(new_model)

def copy_chain_with_renumbering(chain, new_chain_id, coord_offset):
    """
    Copy a chain, renumber residues starting at 1,
    and translate coordinates by coord_offset.
    """
    new_chain = Chain.Chain(new_chain_id)

    new_resi = 1
    for res in chain.get_residues():
        # keep hetero flag & insertion code
        hetflag, _, icode = res.id
        new_res = Residue(
            id=(hetflag, new_resi, icode),
            resname=res.resname,
            segid=res.segid
        )

        for atom in res.get_atoms():
            new_atom = Atom(
                atom.name,
                atom.coord + coord_offset,
                atom.bfactor,
                atom.occupancy,
                atom.altloc,
                atom.fullname,
                atom.serial_number,
                element=atom.element
            )
            new_res.add(new_atom)

        new_chain.add(new_res)
        new_resi += 1

    return new_chain

# Duplicate + rename + renumber
for cmap, off in zip(maps, offsets):
    for old_id in input_chain_order:
        orig_chain = orig_chains[old_id]
        new_chain_id = cmap[old_id]

        new_chain = copy_chain_with_renumbering(
            orig_chain,
            new_chain_id,
            off
        )

        new_model.add(new_chain)

io = PDBIO()
io.set_structure(new_struct)
io.save(out_pdb)

print("Wrote:", out_pdb)
