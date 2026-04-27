import pandas as pd

aa_dict = ['A','V', 'L', 'I', 'M', 'S', 'T', 'C', 'P', 'N', 'Q', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E','G']

mutant_positionsa = []
#----------3436--41---46---51----56---61---66---71---76---81---86---91---96---101-111--116--121--126--131--136--141--146--151--156--161--166--171--176--181--186--191--196--201--206--211--216--221--226--231--236--241--246--251--256--261--266--271--276--281--286--291--296--301--306
wt_seqa = 'VVTPPGPELVLNVSSTFVLTCSGSAPVVWERMSQEPPQEMAKAQDGTFSSVLTLTNLTGLDTGEYFCTHNDDERKRLYIFVPDPTVGFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE'
pos_maska='---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
for i, v in enumerate(pos_maska):
    if v == '#':
        mutant_positionsa.append(i + 34)

print(mutant_positionsa)
#------------3436--41---46---51----56---61---66---71---76---81---86---91---96---101-111--116--121--126--131--136--141--146--151--156--161--166--171--176--181--186--191--196--201--206--211--216--221--226--231--236--241--246--251--256--261--266--271--276--281--286--291--296--301--306
wt_seqb   = 'VVTPPGPELVLNVSSTFVLTCSGSAPVVWERMSQEPPQEMAKAQDGTFSSVLTLTNLTGLDTGEYFCTHNDDERKRLYIFVPDPTVGFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE'
pos_maskb = '---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------'
mutant_positionsb = []
for i, v in enumerate(pos_maskb):
    if v == '#': mutant_positionsb.append(i+34)

print(mutant_positionsb)

tasks = []

#Chain A mutations
for pos in mutant_positionsa:
    wt_aa = wt_seqa[pos-34]
    for aa in aa_dict:
        if wt_aa != 'C':
            tasks.append((f'X {pos}',aa))

#Chain B mutations
for pos in mutant_positionsb:
    wt_aa = wt_seqb[pos - 34]
    for aa in aa_dict:
        if wt_aa != 'C':
            tasks.append((f'Y {pos}',aa))

print(tasks)


df = pd.DataFrame(tasks, columns=["pdb_pos", "mut_aa"])


df.to_csv("/weka/scratch/jgray21/zyhuggan/for_zy/csv_files/mutant_list.csv", index=False)