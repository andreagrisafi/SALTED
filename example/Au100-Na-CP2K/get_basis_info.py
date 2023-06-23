import os
import sys
import math
import numpy as np
from ase.io import read
from itertools import islice
import copy
import argparse
import ctypes
import time

sys.path.insert(0, './')
import inp

species = inp.species

npgf = {}
lmax = {}
nmax = {}
alphas = {}
contra = {}
fbasis = open("new_basis_entry",'w+')
fbasis.write('   if basis=="'+inp.dfbasis+'":\n\n')
for spe in species:
    lmaxlist = [] 
    alphalist = {}
    for l in range(10):
        nmax[(spe,l)] = 0 
        alphalist[l] = [] 
    with open(spe+"-"+inp.dfbasis) as f:
         for line in f:
             nsets = int(list(islice(f, 1))[0])
             for iset in range(nsets):
                 line = list(islice(f, 1))[0]
                 llmin = int(line.split()[1])
                 llmax = int(line.split()[2])
                 nnpgf = int(line.split()[3])
                 nmaxtemp = {}
                 for l in range(llmin,llmax+1):
                     lmaxlist.append(l)
                     nmaxtemp[l] = int(line.split()[4+l-llmin])
                     nmax[(spe,l)] += nmaxtemp[l]
                     alphalist[l].append(np.zeros(nnpgf)) 
                     contra[(spe,l)] = np.zeros((nmaxtemp[l],nnpgf))
                 lines = list(islice(f, nnpgf))
                 for ipgf in range(nnpgf):
                     line = lines[ipgf].split()
                     alpha = float(line[0])
                     icount = 0
                     for l in range(llmin,llmax+1):
                         alphalist[l][-1][ipgf] = alpha
                         for n in range(nmaxtemp[l]):
                             contra[(spe,l)][n,ipgf] = line[1+icount]
                             icount += 1
             break
    lmax[spe] = max(lmaxlist)
    print("L_max = ", lmax[spe])
    for l in range(lmax[spe]+1):
        alphas[(spe,l)] = np.array(alphalist[l]).flatten() 
        np.savetxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat",alphas[(spe,l)])
        #np.savetxt(spe+"-"+inp.dfbasis+"-contraction-coeffs-L"+str(l)+".dat",contra[(spe,l)])

    # SAVE NEW BASIS AS A SALTED DICTIONARY 
    fbasis.write('      lmax["'+spe+'"] = '+str(lmax[spe])+'\n')
    fbasis.write('\n')
    for l in range(lmax[spe]+1):
        fbasis.write('      nmax[("'+spe+'",'+str(l)+')] = '+str(nmax[(spe,l)])+'\n')
    fbasis.write('\n')

fbasis.write('      return [lmax,nmax]\n')
fbasis.write('\n')
fbasis.close()
