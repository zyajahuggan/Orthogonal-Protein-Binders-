import pandas as pd

aa_dict = ['A','V', 'L', 'I', 'M', 'S', 'T', 'C', 'P', 'N', 'Q', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E','G']

mutant_positionsa = []
#----------6----11---16---21---26---31---36---41---46---51---56---61---66---71---76---81---86---91---96---101
wt_seqa = 'TIAEPAMIAECKTRTEVFEISRRLIDRTNANFLVWPPCVEVQRCSGCCNNRNVQCRPTQVQLRPVQVRKIEIVRKKPIFKKATVTLEDHLACKCETV'
pos_maska='---------------#####################-----------------------------#################---------------'
for i, v in enumerate(pos_maska):
    if v == '#':
        #mutant_positionsa.append(i)
        mutant_positionsa.append(i + 6)

print(mutant_positionsa)

wt_seqb   = 'TIAEPAMIAECKTRTEVFEISRRLIDRTNANFLVWPPCVEVQRCSGCCNNRNVQCRPTQVQLRPVQVRKIEIVRKKPIFKKATVTLEDHLACKCETV'
pos_maskb = '-----#######------------------------------------######--------------------------------------#####'
mutant_positionsb = []
for i, v in enumerate(pos_maskb):
    if v == '#': mutant_positionsb.append(i+6)

print(mutant_positionsb)

tasks = []

#Chain A mutations
for pos in mutant_positionsa:
    wt_aa = wt_seqa[pos-6]
    for aa in aa_dict:
        if wt_aa != 'C':
            tasks.append((f'A {pos}',aa))

#Chain B mutations
for pos in mutant_positionsb:
    wt_aa = wt_seqb[pos - 6]
    for aa in aa_dict:
        if wt_aa != 'C':
            tasks.append((f'B {pos}',aa))

print(tasks)


df = pd.DataFrame(tasks, columns=["pdb_pos", "mut_aa"])


df.to_csv("/weka/scratch/jgray21/zyhuggan/for_zy/csv_files/updatedmutant_list.csv", index=False)