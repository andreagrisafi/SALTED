import os
import argparse
import os.path as osp
import sys
import time

import numpy as np

from salted.sys_utils import ParseConfig, read_system, sort_grid_data

def build():
    # load prediction dataset
    inp = ParseConfig().parse_input()
    def add_command_line_arguments_contraction():
        parser = argparse.ArgumentParser()
        parser.add_argument("-vl", "--validation", action='store_true', help="Move SALTED-predicted coefficients for the validations into the relevant AIMS data folders")
        args = parser.parse_args()
        return args

    args = add_command_line_arguments_contraction()
    validation = args.validation
    ntrain = int(inp.gpr.trainfrac*inp.gpr.Ntrain)

    if validation:
        spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
        dirname = osp.join(inp.qm.path2qm, 'data')
        rdir = f"regrdir_{inp.salted.saltedname}"
        trainrangetot = np.loadtxt(osp.join(
            inp.salted.saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"
        ), int)
        testset = np.setdiff1d(list(range(ndata)),trainrangetot)
        g = open('validation_ml_maes', 'w+')

    else:
        spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(
            filename = inp.prediction.filename,
            spelist = inp.system.species,
            dfbasis = inp.qm.dfbasis,
        )
        dirname = osp.join(inp.qm.path2qm, inp.prediction.predict_data)
        testset = range(ndata)
        g = open('ml_maes', 'w+')
    
    testset = [x+1 for x in testset]
    start_time = time.time()
    av_err = 0
    errs = []
    
    for i in testset:
        dirn = osp.join(dirname, str(i))
        # f = open(dirn+'rho_scf.out')
        # r_con = [float(line.split()[-1]) for line in f]
        # f = open(dirn+'rho_df.out')
        # r_ri = [float(line.split()[-1]) for line in f]
        # f = open(dirn+'partition_tab.out')
        # part = [float(line.split()[-1]) for line in f]

        r_con = np.loadtxt(osp.join(dirn, 'rho_scf.out'))
        r_ri = np.loadtxt(osp.join(dirn, 'rho_ml.out'))
        part = np.loadtxt(osp.join(dirn, 'partition_tab.out'))
        # r_con.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        # r_ri.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        # part.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        r_con = sort_grid_data(r_con)
        r_ri = sort_grid_data(r_ri)
        part = sort_grid_data(part)

        err = np.abs(r_ri[:,3]-r_con[:,3])
        norm = np.dot(r_con[:,3],part[:,3])
        int_err = np.dot(err,part[:,3])*100/norm
        errs.append(int_err)
        g.write(str(i)+'    '+str(int_err)+'\n')
        g.flush()

    g.close()
    av_err = np.average(errs)
    # sem = np.std(errs)/np.sqrt(ndata)

    print('% MAE =', av_err)
    end_time = time.time()
    print(f"time_cost = {end_time - start_time:.2f} s")

if __name__ == "__main__":
    build()
