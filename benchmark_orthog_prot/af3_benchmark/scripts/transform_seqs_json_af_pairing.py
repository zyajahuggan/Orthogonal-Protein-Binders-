import re
import json
import os


"""
This will be able to take in orthogonal pairs and output json files that will match cognate and non cognate matches for all of them
a will 
a will be the cognation match with b 
c will be the cognate match with d 
so we will want to create json files for a:b (cognate) and a:d non cognate
c:d cognate c:b non cognate 
"""

fasta = """
>3QBV_1|Chains A, C|Cell division control protein 42 homolog|Homo sapiens (9606)
MQTIKCVVVGDGAVGKTCLLISYTTNKFPSEYVPTVFDNYAVTVMIGGEPYTLGLRDTAGQEDYDRLRPLSYPQTDVFLVCFSVVSPSSFENVKEKWVPEITHHCPKTPFLLVGTQIDLRDDPSTIEKLAKNKQKPITPETAEKLARDLKAVKYVECSALTQKGLKNVFDEAILAALE
>3QBV_2|Chains B, D|Intersectin-1|Homo sapiens (9606)
DMLTPTERKRQGYIHELIVTEENYVNDLQLVTEIFQKPLMESELLTEKEVAMIFVNWKELIMCNIKLLKALRVRKKMSGEKMPVKMIGDILSAQLPHMQPYIRFCSRQLNGAALIQQKTDEAPDFKEFVKRLAMDPRCKGMPLSEFILKPMQRVTRYPLIIKNILENTPENHPDHSHLKHALEKAEELCSQVNEGVREKENSDRLEWIQAHVQCEGLSEQLVFNSVTNCLGPRKFLHSGKLYKAKSNKELYGFLFNDFLLLTQITKPLGSSGTDKVFSPKSNLQYKMYKTPIFLNEVLVKLPTDPSGDEPIFHISHIDRVYTLRAESINERTAWVQKIKAASELYIETEKK
>1KI1_1|Chains A, C|G25K GTP-binding protein, placental isoform|Homo sapiens (9606)
MQTIKCVVVGDGAVGKTCLLISYTTNKFPSEYVPTVFDNYAVTVMIGGEPYTLGLFDTAGQEDYDRLRPLSYPQTDVFLVCFSVVSPSSFENVKEKWVPEITHHCPKTPFLLVGTQIDLRDDPSTIEKLAKNKQKPITPETAEKLARDLKAVKYVECSALTQKGLKNVFDEAILAALEPPEPKKSRRS
>1KI1_2|Chains B, D|intersectin long form|Homo sapiens (9606)
DMLTPTERKRQGYIHELIVTEENYVNDLQLVTEIFQKPLMESELLTEKEVAMIFVNWKELIMCNIKLLKALRVRKKMSGEKMPVKMIGDILSAQLPHMQPYIRFCSRQLNGAALIQQKTDEAPDFKEFVKRLEMDPRCKGMPLSSFILKPMQRVTRYPLIIKNILENTPENHPDHSHLKHALEKAEELCSQVNEGVREKENSDRLEWIQAHVQCEGLSEQLVFNSVTNCLGPRKFLHSGKLYKAKNNKELYGFLFNDFLLLTQITKPLGSSGTDKVFSPKSNLQYMYKTPIFLNEVLVKLPTDPSGDEPIFHISHIDRVYTLRAESINERTAWVQKIKAASELYIETEKKKR
>1EMV_1|Chain A|IMMUNITY PROTEIN IM9|Escherichia coli (562)
MELKHSISDYTEAEFLQLVTTICNADTSSEEELVKLVTHFEEMTEHPSGSDLIYYPKEGDDDSPSGIVNTVKQWRAANGKSGFKQG
>1EMV_2|Chain B|COLICIN E9|Escherichia coli (562)
MESKRNKPGKATGKGKPVGDKWLDDAGKDSGAPIPDRIADKLRDKEFKSFDDFRKAVWEEVSKDPELSKNLNPSNKSSVSKGYSPFTPKNQQVGGRKVYELHHDKPISQGGEVYDMDNIRVTTPKRHIDIHRGK
>7CEI_1|Chain A|PROTEIN (COLICIN E7 IMMUNITY PROTEIN)|Escherichia coli str. K12 substr. (316407)
MELKNSISDYTEAEFVQLLKEIEKENVAATDDVLDVLLEHFVKITEHPDGTDLIYYPSDNRDDSPEGIVKEIKEWRAANGKPGFKQG
>7CEI_2|Chain B|PROTEIN (COLICIN E7 IMMUNITY PROTEIN)|Escherichia coli str. K12 substr. (316407)
ERFAREPMAAGHRMWQMAGLKAQRAQTDVNNKKAAFDAAAKEKSDADVALSSALERRKQKENKEKDAKAKLDKESKRNKPGKATGKGKPVNNKWLNNAGKDLGSPVPDRIANKLRDKEFKSFDDFRKKFWEEVSKDPELSKQFSRNNNDRMKVGKAPKTRTQDVSGKRTSFELHHEKPISQNGGVYDMDNISVVTPKRHIDIHRGK
"""

def parse_fasta(text):
    sequences = []
    header, seq = None, []
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if header:
                sequences.append([header, "".join(seq)])
            header, seq = line[1:], []
        else:
            seq.append(line)
    if header:
        sequences.append([header, "".join(seq)])
    return sequences

labels = ["a", "as", "b", "bs"]
chain_as, chain_bs = [], []

for i, (_, seq) in enumerate(parse_fasta(fasta)):
    system = (i // 4) + 1
    label = labels[i % 4]
    name = f"{system}{label}"
    if "s" in label:
        chain_bs.append([name, seq])
    else:
        chain_as.append([name, seq])
        
SEEDS = list(range(1, 250))
OUTPUT_DIR = "../inputs/cross_matching_orthogonal_prot"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_word_containing_number(text, number):
    pattern = rf'\b\w*(?<!\d){number}(?!\d)\w*\b'
    return re.findall(pattern, text)

def extract_number(name):
    match = re.search(r'\d+', name)
    return match.group() if match else None

total = 0

for chain_a in chain_as:
    receptor_number = extract_number(chain_a[0])

    for chain_b in chain_bs:
        ligand_name = chain_b[0]
        matches = find_word_containing_number(ligand_name, receptor_number)

        if matches:
            name = f"{chain_a[0]}_vs_{chain_b[0]}"
            sequences = [
                {"protein": {"id": ["A"], "sequence":chain_a[1]}},
                {"protein": {"id": ["B"], "sequence": chain_b[1]}},
            ]
            job = {
                "name": name,
                "dialect": "alphafold3",
                "version": 1,
                "sequences": sequences,
                "modelSeeds": SEEDS,
            }
            with open(f"{OUTPUT_DIR}/{name}.json", "w") as f:
                json.dump(job, f, indent=4)

            print(f"Created: {name}.json")
            total += 1

print(f"\nTotal jobs: {total}")