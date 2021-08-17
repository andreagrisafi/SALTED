import os
import sys
import numpy as np
import time
from ase.io import read

sys.path.insert(0, './')
import inp

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

for iconf in xrange(ndata):
    print iconf+1
    
    # compute coefficients
    Proj = np.load(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy")
    Over = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    invsqrtdiag = 1.0/np.sqrt(np.diagonal(Over))
    Proj = Proj * invsqrtdiag
    ovlpnorm = np.outer(invsqrtdiag,invsqrtdiag)
    Over = Over * ovlpnorm
    Coef = np.linalg.solve(Over,Proj)
    np.save(inp.path2qm+"coefs_normalized/coefficients_conf"+str(iconf)+".npy",Coef)
    np.save(inp.path2qm+"projs_normalized/projections_conf"+str(iconf)+".npy",Proj)
    np.save(inp.path2qm+"ovlps_normalized/overlap_conf"+str(iconf)+".npy",Over)
