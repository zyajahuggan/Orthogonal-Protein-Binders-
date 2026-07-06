# Running AlphaFold3 on the DSAI Cluster

A step-by-step guide for new lab members.

---

## Before You Start

**What AlphaFold3 does:** Given amino acid sequences, AlphaFold3 predicts the 3D structure of a protein or protein complex. It first searches large sequence databases to build a multiple sequence alignment (MSA) — an evolutionary context for each chain — then uses a deep learning model to predict coordinates and confidence scores.

**What you need before running:**
- [ ] Your sequence(s) as a PDB file or as raw amino acid sequences (one-letter code)
- [ ] A project directory on `/weka/scratch/jgray21/`
- [ ] A SLURM account on the DSAI cluster

**How AF3 is installed on this cluster:** AlphaFold3 is *not* available as a loadable module. Instead, use the wrapper script directly. The wrapper launches AF3 inside a Singularity container and handles all setup for you — you do not need to load any modules yourself.

### Cluster paths — copy these exactly, do not change them

| Resource | Path |
|----------|------|
| Wrapper script | `/weka/apps/software/extern/singularity/alphafold/run_alphafold3.sh` |
| Model parameters | `/apps/software/extern/singularity/alphafold/models` |
| Sequence databases | `/apps/software/extern/singularity/alphafold/databases` |

These are already filled in for you in `config.sh` and all scripts in this guide.

### Two things every AF3 run needs

1. An **input JSON file** that describes your sequences (you'll create this in Step 1)
2. A **SLURM GPU job** to submit the run (scripts are in `scripts/`)

AF3 cannot be run interactively — it must go through SLURM.

---

## Step 1: Create Your Input JSON

AF3 takes a JSON file as input, not a PDB file directly. The JSON tells AF3 what sequences to fold and how to handle each chain.

### What a minimal input JSON looks like

Here is an example for a two-chain complex (an antibody VHH bound to a receptor):

```json
{
    "name": "my_complex",
    "dialect": "alphafold3",
    "version": 1,
    "sequences": [
        {
            "protein": {
                "id": ["A"],
                "sequence": "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR"
            }
        },
        {
            "protein": {
                "id": ["B"],
                "sequence": "REENANFNKIFLPTIYSIIFLTGIVGNGLVILVMGYQKKLRSMTDKYRLHLSVADLLFVITLPFW"
            }
        }
    ],
    "modelSeeds": [1]
}
```

**Key fields:**
- `"name"` — used to name the output directory. **AF3 lowercases this**, so `"My_Complex"` creates output at `my_complex/`. Keep it lowercase with underscores.
- `"id"` — chain identifier letter(s). Use single uppercase letters.
- `"sequence"` — one-letter amino acid sequence, no spaces or newlines.
- `"modelSeeds"` — list of random seeds. `[1]` gives one prediction; `[1, 2, 3]` gives three.

### Controlling whether AF3 searches the database for each chain

This is the most important thing to understand about the input format:

| `unpairedMsa` field in JSON | What AF3 does |
|-----------------------------|---------------|
| **Field absent** (not present at all) | AF3 runs a database search (jackHMMer/HHblits) for that chain — this is the default and what you want most of the time |
| `"unpairedMsa": ""` (empty string) | AF3 skips the database search for that chain and uses no MSA |

**When to skip the MSA:** If a chain has no natural homologs (e.g. a de novo designed loop with random sequence), or if chain lengths vary across many inputs and you cannot reuse a single alignment, setting `"unpairedMsa": ""` is appropriate. For most real proteins, leave the field absent so AF3 searches the database.

### Generate your JSON automatically from a PDB file

If you have a PDB file with the structure (or a template with the right chains), use the provided script:

```bash
conda activate /weka/scratch/jgray21/amontan6/envs/analysis

python3 /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/prepare_input_json.py \
    --pdb my_structure.pdb \
    --chains A B \
    --name my_complex \
    --output my_complex.json
```

To skip the MSA for a specific chain (e.g. chain B):

```bash
python3 .../prepare_input_json.py \
    --pdb my_structure.pdb \
    --chains A B \
    --skip_msa_chains B \
    --name my_complex \
    --output my_complex.json
```

**Verify your JSON before submitting:**

```bash
python3 -c "
import json
with open('my_complex.json') as f:
    d = json.load(f)
for s in d['sequences']:
    p = s['protein']
    has_msa_field = 'unpairedMsa' in p
    msa_status = 'SKIP MSA' if has_msa_field else 'will search DB'
    print(f'Chain {p[\"id\"][0]}: {len(p[\"sequence\"])} aa  [{msa_status}]')
"
```

---

## Step 2: Choose Your Workflow

```
How many complexes do I need to predict?
│
├── 1–10 structures
│   └── Workflow A: one job per complex (start here if you're new)
│
├── 10–200 structures
│   └── Workflow B: array job, each design gets its own database search
│
└── 200+ structures with conserved chains (e.g. same receptor, designed binders)
    └── Workflow C: shared-MSA array — advanced, ask the lab before using
```

---

## Workflow A — One Complex at a Time

> Use this if you're running 1–10 structures or are new to AF3. It's the simplest option and always correct.

**Step A1.** Copy and edit the single-run script:

```bash
cp /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/run_af3_single.sh \
   /path/to/your/project/scripts/run_af3_single.sh
```

Open the script and set the three variables at the top:

```bash
INPUT_DIR=/path/to/your/project/af3_inputs   # directory containing your input JSON
INPUT_FILE=my_complex.json                    # just the filename, not the full path
OUTPUT_DIR=/path/to/your/project/af3_outputs  # where predictions will be saved
```

**Step A2.** Create your input directory and put your JSON there:

```bash
mkdir -p /path/to/your/project/af3_inputs
cp my_complex.json /path/to/your/project/af3_inputs/
```

**Step A3.** Submit:

```bash
mkdir -p /path/to/your/project/logs
sbatch /path/to/your/project/scripts/run_af3_single.sh
```

**Step A4.** Watch the job:

```bash
squeue -u $USER          # shows running/pending jobs
# Your job is done when it disappears from squeue

# Check the log:
cat /path/to/your/project/logs/af3_single_<JOBID>.out
```

**What success looks like in the log:**
```
Started:    Thu Jun 26 14:23:01 EDT 2026
Input:      /path/to/your/project/af3_inputs/my_complex.json
...
Finished:   Thu Jun 26 18:41:17 EDT 2026
```

**Expected time:** 4–12 hours on an A100 for a typical 2–4 chain complex.

---

## Workflow B — Many Complexes, One MSA Per Design

> Use this when you have 10–200 structures and want each to get its own independent database search. This is correct by default and requires no special assumptions about your sequences.

**Step B1.** Prepare one input JSON per design. If all your structures are PDB files in one directory, generate JSONs in a loop:

```bash
conda activate /weka/scratch/jgray21/amontan6/envs/analysis

for pdb in /path/to/pdbs/*.pdb; do
    name=$(basename "$pdb" .pdb)
    python3 /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/prepare_input_json.py \
        --pdb "$pdb" \
        --chains A B \
        --name "$name" \
        --output /path/to/your/project/af3_inputs/"${name}.json"
done
```

**Step B2.** Count your inputs — you'll need this for the array size:

```bash
ls /path/to/your/project/af3_inputs/*.json | wc -l
# e.g. output: 87
```

**Step B3.** Copy and edit the array script:

```bash
cp /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/run_af3_full_array.sh \
   /path/to/your/project/scripts/run_af3_full_array.sh
```

Set the three variables at the top, and update the `--array` line to match your count (if you have 87 files, use `--array=0-86%20`):

```bash
INPUT_DIR=/path/to/your/project/af3_inputs
OUTPUT_DIR=/path/to/your/project/af3_outputs
# In the #SBATCH header:
#SBATCH --array=0-86%20    # 87 files, 20 running at once
```

**Step B4.** Submit:

```bash
sbatch /path/to/your/project/scripts/run_af3_full_array.sh
```

**Step B5.** Monitor array progress:

```bash
squeue -u $USER            # see all your running tasks
squeue -u $USER | grep af3 # filter to AF3 jobs only

# Count completed output directories:
ls /path/to/your/project/af3_outputs/ | wc -l
```

**Note on concurrency:** The `%20` in `--array=0-86%20` limits 20 tasks to run simultaneously. For longer jobs (>4 h each), keep this at 20–30. For shorter jobs (<2 h), you can raise it to 50.

---

## Workflow C — Shared MSA (Advanced, Large Scale)

> Use this only when you have **200+ structures** where at least one chain is identical (or nearly identical) across all inputs, like the same receptor sequence or the same antibody scaffold. It avoids running the database search thousands of times by computing it once and reusing it.

**When it's valid:**
- Chain is identical across all designs (e.g. a receptor target)
- Chain has a conserved framework (e.g. VHH nanobodies sharing the same 130-aa scaffold)

**When NOT to use it:**
- Sequences are highly diverse — each design should get its own MSA
- You're not sure — ask the lab first

**How it works:**
1. Pick one representative structure and run only the MSA search stage for it (CPU only)
2. Graft the resulting MSA onto all other structures (just swapping the query sequence)
3. Run GPU inference on all structures using the pre-computed MSA

**Step C1.** Create a representative input JSON for the MSA stage:

```bash
python3 /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/prepare_input_json.py \
    --pdb representative.pdb \
    --chains A B \
    --name my_project_representative \
    --output af3_msa_input/representative.json
```

**Step C2.** Run MSA search only (CPU, no GPU needed):

```bash
cp /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/run_af3_msa_stage.sh \
   scripts/run_af3_msa_stage.sh
# Edit INPUT_DIR / OUTPUT_DIR at the top, then:
sbatch scripts/run_af3_msa_stage.sh
```

This takes 2–8 hours. When it finishes you'll have a `*_data.json` file (~38 MB) in your MSA output directory with all alignments embedded.

**Step C3.** Graft the representative MSA onto all member structures:

```bash
python3 /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/prepare_member_jsons.py \
    --rep_data_json af3_msa_output/my_project_representative/my_project_representative_data.json \
    --pdb_dir /path/to/your/pdbs \
    --chains_to_reuse A B \
    --output_dir af3_gpu_inputs/
```

Use `--chains_skip_msa` for any chain that should keep an empty MSA (e.g. if lengths vary across designs).

**Step C4.** Run GPU inference array (reads pre-computed MSAs, no database search):

```bash
cp /weka/scratch/jgray21/amontan6/software/af3_guide/scripts/run_af3_gpu_array.sh \
   scripts/run_af3_gpu_array.sh
# Edit INPUT_DIR / OUTPUT_DIR and array size, then:
sbatch scripts/run_af3_gpu_array.sh
```

**Real example:** This workflow was used for 1,530 designed antibody–receptor complexes in the lab's deltaN-term project. Running MSA searches for all 1,530 structures would have taken weeks; sharing one representative MSA reduced it to a single 4-hour CPU job.

---

## Reading Your Results

### Output directory layout

After a successful run, your output directory will contain one subdirectory per prediction named after the `"name"` field in your input JSON (lowercased):

```
af3_outputs/
└── my_complex/
    ├── my_complex_model.cif           ← best-ranked structure (CIF format)
    ├── my_complex_confidences.json    ← per-residue pLDDT and PAE for best model
    ├── my_complex_summary_confidences.json  ← overall scores for best model
    ├── ranking_scores.csv             ← scores for all 5 samples
    ├── seed-1_sample-0/        ...