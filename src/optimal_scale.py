import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
import argparse
import random

import basis

sys.path.insert(0, './')
import inp


# read species
spelist = inp.species

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

errors = np.loadtxt("errors_multiscale.dat")
rcuts = [2.0,3.0,4.0,5.0,6.0]

f = open("errors_optimal.dat","w")
orcuts = []
i = 0
for spe in spelist:
    for l in xrange(lmax[spe]+1):
        for n in xrange(nmax[(spe,l)]):
            imin = np.argmin(errors[i])
            orcuts.append(int(rcuts[imin]))
            print >> f, ""
            print >> f, "spe:",spe,"L:",l,"n:",n, "rcut:", rcuts[imin]
            print >> f, "--------------------------------------"
            print >> f, "% RMSE =", errors[i,imin]
            i += 1
f.close()
np.savetxt("optimal_rcuts.dat",np.array(orcuts),fmt="%i")
