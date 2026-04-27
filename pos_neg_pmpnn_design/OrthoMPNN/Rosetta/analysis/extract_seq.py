from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

parser = PDBParser(QUIET=True)
structure = parser.get_structure("prot", "/home/zhuggan1/scr4_jgray21/zhuggan1/projects/orthosystems/tryingggg/pos_neg_design/OrthoMPNN/Rosetta/output_pdbs/relaxed_ortho_5.pdb")

for chain in structure.get_chains():
    seq = ""
    for res in chain:
        if res.has_id("CA"):
            seq += seq1(res.get_resname())
    print(chain.id, seq)
