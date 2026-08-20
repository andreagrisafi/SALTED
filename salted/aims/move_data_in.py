import os
import argparse
import sys
import os.path as osp

import numpy as np

from salted.sys_utils import ParseConfig, detect_mpi, distribute_jobs, read_system


def build():
    inp = ParseConfig().parse_input()
    saltedname = inp.salted.saltedname

    comm, size, rank, parallel = detect_mpi()

    def add_command_line_arguments_contraction():
        parser = argparse.ArgumentParser()
        parser.add_argument("-vl", "--validation", action='store_true', help="Move SALTED-predicted coefficients for the validations into the relevant AIMS data folders")
        args = parser.parse_args()
        return args

    args = add_command_line_arguments_contraction()
    validation = args.validation
    ntrain = int(inp.gpr.trainfrac*inp.gpr.Ntrain)
    
    if rank == 0: print("WARNING! This script assumes you will use an AIMS version >= 240403 to read the predicted RI coefficients. If this is not true, please use move_data_in_reorder instead.")

    if validation:
        species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
        pdir = f"validations_{saltedname}"
        rdir = f"regrdir_{saltedname}"

        # define validation set
        trainrangetot = np.loadtxt(osp.join(
            inp.salted.saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"
        ), int)
        structure_list = np.setdiff1d(list(range(ndata)),trainrangetot)
        np.savetxt(osp.join(
            inp.salted.saltedpath, f"temp_validation_set.txt"), structure_list,fmt='%i'
        )
        datadir = 'data'

    else:
        species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(filename=inp.prediction.filename,spelist = inp.system.species, dfbasis = inp.qm.dfbasis)
        pdir = f"predictions_{saltedname}_{inp.prediction.predname}"
        structure_list = list(range(ndata))
        datadir = inp.prediction.predict_data
    
    # Distribute structures to tasks
    if parallel:
        conf_range = distribute_jobs(comm, structure_list)
    else:
        conf_range = structure_list
    
    for i in conf_range:
        if inp.salted.verbose:
            print(f"processing {i+1}/{ndata} frame")
        t = np.loadtxt(os.path.join(
            inp.salted.saltedpath, pdir,
            f"M{inp.gpr.Menv}_zeta{inp.gpr.z}", f"N{ntrain}_reg{int(np.log10(inp.gpr.regul))}",
            f"COEFFS-{i+1}.dat",
        ))
        n = len(t)
    
        dirpath = os.path.join(inp.qm.path2qm, datadir, f"{i+1}")
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
    
        np.savetxt(os.path.join(dirpath, f"ri_restart_coeffs_predicted.out"), t)

if __name__ == "__main__":
    build()
