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

from salted.sys_utils import ParseConfig
inp = ParseConfig().parse_input()

species = inp.system.species

print("Reading AOs info...")
laomax = {}
naomax = {}
npgf = {}
aoalphas = {}
contra = {}
lmax = {}
nmax = {}
for spe in species:
    with open("BASIS_MOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == inp.qm.qmbasis:
                line = list(islice(f, 2))[1]
                laomax[spe] = int(line.split()[2])
                npgf[spe] = int(line.split()[3])
                for l in range(laomax[spe]+1):
                    naomax[(spe,l)] = int(line.split()[4+l])
                    contra[(spe,l)] = np.zeros((naomax[(spe,l)],npgf[spe]))
                lines = list(islice(f, npgf[spe]))
                aoalphas[spe] = np.zeros(npgf[spe])
                for ipgf in range(npgf[spe]):
                    line = lines[ipgf].split()
                    aoalphas[spe][ipgf] = float(line[0])
                    icount = 0
                    for l in range(laomax[spe]+1):
                        for n in range(naomax[(spe,l)]):
                            contra[(spe,l)][n,ipgf] = line[1+icount]
                            icount += 1  
                break
    nalphas = npgf[spe]*3
    alphamin = min(aoalphas[spe])
    alphamax = max(aoalphas[spe])
    alphamin -= alphamin/3
    alphamax += alphamax/3
    r=(alphamax/alphamin)**(1.0/float(nalphas-1))
    alphas=np.zeros(nalphas)
    for i in range(nalphas):
        alphas[i] = 2*alphamin*r**i
    np.savetxt("alphas-"+str(spe)+".txt",-np.sort(-alphas))
