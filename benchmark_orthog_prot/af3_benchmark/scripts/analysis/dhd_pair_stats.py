"""Evaluates whether two cognate pairs of orthogonal proteins in dhd data set score above all of those two non cognate
distractions. Im goin to treath this like cross docking even though this is a set of six heterodimers that are all 
orthogonal to the other. so if there are pairs a,b,c,d,e,f. I will compare a to b, a to c, a to d .... then i will 
b to d and so on and calulate the frequency that the lowest score of the cognate pair is above the highest of the 
non-cognate"""