import numpy as np
import sys

species = ["H","O"]
lmax = {}
nmax = {}

ispe = 0
for spe in species:
    print("species:", spe)
    f = open(spe+"-LRI-DZVP-MOLOPT-GTH-MEDIUM_info.txt","r")
    lines = f.readlines()
    f.close()
    nbas = lines[0]
    print("total number of functions = ", nbas)
    lvals = np.asarray(lines[2:],int)
    lmax[spe] = max(lvals)
    print("lmax =",lmax[spe])
    for l in range(lmax[spe]+1):
        nmax[(spe,l)] = 0
    for l in lvals:
        nmax[(spe,l)] += 1
    for l in range(lmax[spe]+1):
        print("l =",l,"nmax =",nmax[(spe,l)])
    
        
