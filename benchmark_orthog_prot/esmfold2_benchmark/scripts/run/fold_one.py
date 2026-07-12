import sys, os, json
import numpy as np
from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    ProteinInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

def parse_fasta(path):
    entries = []
    with open(path) as f:
        name, seq = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    entries.append((name, "".join(seq)))
                name = line[1:]
                seq = []
            else:
                seq.append(line)
        if name:
            entries.append((name, "".join(seq)))
    return entries

task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
entries = parse_fasta("/home/cbufffor1/scratchjgray21/cbufford1/projects/orthogonal_protein_benchmark/Orthogonal-Protein-Binders-/benchmark_orthog_prot/esmfold2_benchmark/inputs/cross_docking/cross_docking.fasta")
job_name, sequence = entries[task_id]

outfile = f"outputs/{job_name}.cif"
if os.path.exists(outfile):
    print(f"Skipping {job_name}, already done")
    sys.exit(0)

print("Loading ESMFold2 model...")
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").cuda().eval()

chains = sequence.split(":")
spi = StructurePredictionInput(
    sequences=[ProteinInput(id=chr(65+i), sequence=s) for i, s in enumerate(chains)]
)

print(f"Folding {job_name} ({len(chains)} chain(s))...")
result = ESMFold2InputBuilder().fold(
    model, spi,
    num_loops=3,
    num_sampling_steps=50,
    num_diffusion_samples=1,
    seed=0,
)

print(f"pLDDT: {float(result.plddt.mean()):.3f}, pTM: {float(result.ptm):.3f}, ipTM: {float(result.iptm):.3f}")

os.makedirs("outputs", exist_ok=True)

# Save structure
with open(outfile, "w") as f:
    f.write(result.complex.to_mmcif())

# Save PAE
pae = result.pae.cpu().numpy()
np.save(f"outputs/{job_name}_pae.npy", pae)

# Save metrics
metrics = {
    "plddt": float(result.plddt.mean()),
    "ptm":   float(result.ptm),
    "iptm":  float(result.iptm),
}
with open(f"outputs/{job_name}_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Done: {job_name} -> {outfile}")