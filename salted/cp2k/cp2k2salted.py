import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
import copy
import time

from salted import basis
from salted.sys_utils import ParseConfig, check_MPI_tasks_count, distribute_jobs

inp = ParseConfig().parse_input()

xyzfile = read(inp.system.filename,":")
ndata = len(xyzfile)
species = inp.system.species
[lmax,nmax] = basis.basiset(inp.qm.dfbasis)

if inp.system.parallel:

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

else:

    comm = None
    size = 1
    rank = 0

if rank==0:

    dirpath = os.path.join(inp.salted.saltedpath, "coefficients")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    
    if inp.salted.saltedtype=="density-response":
        for icart in ["x","y","z"]:
            dirpath = os.path.join(inp.salted.saltedpath, "coefficients", f"{icart}")
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)


if inp.system.parallel:

    comm.Barrier()

    check_MPI_tasks_count(comm, ndata, "configurations")
    conf_range = distribute_jobs(comm, np.arange(ndata,dtype=int))
    print(
        f"Task {rank+1} handles the following configurations: {conf_range}", flush=True
    )

else:

    conf_range = np.arange(ndata,dtype=int)


# init geometry
for iconf in conf_range:

    geom = xyzfile[iconf]
    symbols = geom.get_chemical_symbols()
    natoms = len(symbols)
    # compute basis set size
    nRI = 0
    for iat in range(natoms):
        spe = symbols[iat]
        if spe in species:
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    nRI += 2*l+1

    print("conf", iconf+1, "size =", nRI, flush=True)

    # save overlap matrix in SALTED format
    overlap = np.zeros((nRI, nRI)).astype(np.double)
    for i in range(nRI):
       offset = 4 + i*((nRI+1)*8)
       overlap[:, i] = np.fromfile(os.path.join(
           inp.qm.path2qm, f"conf_{iconf+1}", inp.qm.ovlpfile
        ), dtype=np.float64, offset = offset, count=nRI)

    dirpath = os.path.join(inp.salted.saltedpath, "overlaps")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    np.save(os.path.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{iconf}.npy"), overlap)

    if inp.salted.saltedtype=="density":

        # load density coefficients and check dimension
        coefficients = np.loadtxt(os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", inp.qm.coeffile))
        if len(coefficients)!=nRI:
            print("ERROR: basis set size does not correspond to size of coefficients vector!")
            sys.exit(0)
    
        # save coefficients vector in SALTED format
        #if natoms%2 != 0:
        #    coefficients = np.sum(coefficients,axis=1)
        np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"), coefficients)


    elif inp.salted.saltedtype=="density-response":

        # load density-response coefficients and check dimension
        for icart in ["x","y","z"]:
            # Estimate derivative by finite differences applying an electric field of 0.01V/angs
            coefficients = np.loadtxt(os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", f"{icart}_positive", inp.qm.coeffile))
            coefficients -= np.loadtxt(os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", f"{icart}_negative", inp.qm.coeffile))
            coefficients /= (2*0.0001945) # 0.01 V/angs
            if len(coefficients)!=nRI:
                print("ERROR: basis set size does not correspond to size of coefficients vector!")
                sys.exit(0)

            # save coefficients vector in SALTED format
            np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"{icart}", f"coefficients_conf{iconf}.npy"), coefficients)

    ## save projections vector in SALTED format
    #projections = np.dot(overlap,coefficients)
    #dirpath = os.path.join(inp.salted.saltedpath, "projections")
    #if not os.path.exists(dirpath):
    #    os.mkdir(dirpath)
    #np.save(inp.salted.saltedpath+"projections/projections_conf"+str(iconf)+".npy",projections)
