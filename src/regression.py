import os
import sys
import numpy as np
import time
from scipy import sparse

import basis

sys.path.insert(0, './')
import inp

# sparse-GPR parameters
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

if inp.combo:
    fdir = "rkhs-vectors_"+inp.saltedname+"_"+inp.saltedname2
    rdir = "regrdir_"+inp.saltedname+"_"+inp.saltedname2
else:
    fdir = "rkhs-vectors_"+inp.saltedname
    rdir = "regrdir_"+inp.saltedname

p = sparse.load_npz(inp.saltedpath+fdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf0.npz")
totsize = p.shape[-1]
print("problem dimensionality:", totsize,flush=True)
if totsize>70000:
    print("ERROR: problem dimension too large, minimize directly loss-function instead!")
    sys.exit(0)

# define training set size 
ntrain = int(inp.trainfrac*inp.Ntrain)

# load regression matrices
Avec = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Avec_N"+str(ntrain)+".npy")
Bmat = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/Bmat_N"+str(ntrain)+".npy")

start = time.time()

w = np.linalg.solve(Bmat+np.eye(totsize)*reg,Avec)

print("regression time:", (time.time()-start)/60, "minutes")

np.save(inp.saltedpath+rdir+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",w)
