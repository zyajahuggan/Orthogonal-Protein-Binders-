from Bio.PDB import PDBParser, PPBuilder
import sys

pdb_file = sys.argv[1]

parser = PDBParser()
structure = parser.get_structure("Y", pdb_file)

ppb = PPBuilder()

for model in structure:
    for chain in model:
        seq = ""
        for pp in ppb.build_peptides(chain):
            seq += str(pp.get_sequence())
        print(f"Chain {chain.id}: {seq}")
