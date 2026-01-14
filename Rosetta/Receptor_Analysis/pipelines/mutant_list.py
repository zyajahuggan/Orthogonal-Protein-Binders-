import pandas as pd
import sys
aa_dict = ['A','V', 'L', 'I', 'M', 'S', 'T', 'C', 'P', 'N', 'Q', 'F', 'Y', 'W', 'H', 'K', 'R', 'D', 'E','G']

mutant_positionsy = []
#----------34-----41---46---51---56---61---66---71---76---81---86---91---96---101-------111--116--121--126--131--136--141--146--151--156--161--166--171--176--181--186--191--196--201--206--211--216--221--226--231--236--241--246--251--256--261--266--271--276--281--286--291--296--301--306
wt_seqy = 'VVTPPGPELVLNVSSTFVLTCSGSAPVVWERMSQEPPQEMAKAQDGTFSSVLTLTNLTGLDTGEYFCTHND------DERKRLYIFVPDPTVGFLPNDAEELFIFLTEITEITIPCRVTDPQLVVTLHEKKGDVALPVPYDHQRGFSGIFEDRSYICKTTIGDREVDSDAYYVYRLQVSSINVSVNAVQTVVRQGENITLMCIVIGNEVVNFEWTYPRKESGRLVEPVTDFLLDMPYHIRSILHIPSAELEDSGTYTCNVTESVNDHQDEKAINITVVE'
pos_masky='--------------------------------------------------------------------------------------------------#########---------------------###-------------------####-----------------######------------------------------#####-------------##########-###-----------------------###--------------'
for i, v in enumerate(pos_masky):
    if v == '#':
        mutant_positionsy.append(i + 34)

print(mutant_positionsy)
print(len(mutant_positionsy))
sys.exit()
tasks = []

#Chain Y mutations
for pos in mutant_positionsy:
    wt_aa = wt_seqy[pos-34]
    for aa in aa_dict:
        if wt_aa != 'C':
            tasks.append((f'Y {pos}',aa))

print(tasks)


df = pd.DataFrame(tasks, columns=["pdb_pos", "mut_aa"])


df.to_csv("/scratch4/jgray21/zhuggan1/projects/orthosystems/Receptor_Analysis/pipelines/receptor_mutant_list.csv", index=False)