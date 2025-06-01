import os
import os.path as osp
import sys
import time

import numpy as np

from salted.sys_utils import ParseConfig, read_system, sort_grid_data


def main():
    # load prediction dataset
    inp = ParseConfig().parse_input()
    spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        filename = inp.prediction.filename,
        spelist = inp.system.species,
        dfbasis = inp.qm.dfbasis,
    )

    start_time = time.time()

    dirname = osp.join(inp.qm.path2qm, inp.prediction.predict_data)
    av_err = 0
    errs = np.zeros(ndata)
    g = open('ml_maes', 'w+')
    for i in range(1,ndata+1):
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
        errs[i-1] = int_err
        g.write(str(i)+'    '+str(int_err)+'\n')
        g.flush()

    g.close()
    av_err = np.average(errs)
    # sem = np.std(errs)/np.sqrt(ndata)

    print('% MAE =', av_err)
    end_time = time.time()
    print(f"time_cost = {end_time - start_time:.2f} s")

main()
