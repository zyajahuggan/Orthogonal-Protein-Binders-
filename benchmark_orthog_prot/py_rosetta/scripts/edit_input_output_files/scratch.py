from pathlib import Path
import re



pdb_path = Path("/weka/scratch/jgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/boltz/outputs/dhd/DHD150a_vs_DHD154b/boltz_results_DHD150a_vs_DHD154b/predictions/DHD150a_vs_DHD154b/DHD150a_vs_DHD154b_model_0.cif")


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

def get_cognate_status_dhd(name):
    cognate_status = 0
    pattern = re.compile(r"^(.+)(?:a_vs_\1b|b_vs_\1a)$")
    found = pattern.search(name)
    if found is not None:
        cognate_status = 1
    return cognate_status

name = get_protein_name(pdb_path)
print(name)
print(get_cognate_status_dhd(name))