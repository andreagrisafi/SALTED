import time
import sys
import os.path as osp

import numpy as np
import inp

def sort_grid_data(data:np.ndarray) -> np.ndarray:
    """Sort real space grid data
    The grid data is 2D array with 4 columns (x,y,z,value).
    Sort the grid data in the order of x, y, z.

    Args:
        data (np.ndarray): grid data

    Returns:
        np.ndarray: sorted grid data
    """
    assert data.ndim == 2
    assert data.shape[1] == 4
    data = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]  # last key is primary
    return data


def main():
    from salted.sys_utils import read_system
    spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    start_time = time.time()

    dirname = osp.join(inp.path2qm, 'data')
    av_err = 0
    errs = np.zeros(ndata)
    # g = open('df_maes','w+')
    # for i in range(1,ndata+1):
    g = open('df_maes_test','w+')
    for i in range(1,4):
        dirn = osp.join(dirname, str(i))
        # f = open(dirn+'rho_scf.out')
        # r_con = [float(line.split()[-1]) for line in f]
        # f = open(dirn+'rho_df.out')
        # r_ri = [float(line.split()[-1]) for line in f]
        # f = open(dirn+'partition_tab.out')
        # part = [float(line.split()[-1]) for line in f]

        r_con = np.loadtxt(osp.join(dirn, 'rho_scf.out'))
        r_ri = np.loadtxt(osp.join(dirn, 'rho_df.out'))
        part = np.loadtxt(osp.join(dirn, 'partition_tab.out'))
        # print("before sort by coord")
        # print(i, np.allclose(r_ri[:,:3], r_con[:,:3]), np.allclose(r_ri[:,:3], part[:,:3]))
        # print(i, np.linalg.norm(r_ri[:,:3] - r_con[:,:3]), np.linalg.norm(r_ri[:,:3] - part[:,:3]))
        r_con = sort_grid_data(r_con)
        r_ri = sort_grid_data(r_ri)
        part = sort_grid_data(part)
        # r_con.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        # r_ri.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        # part.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        # print("after sort by coord")
        # print(i, np.allclose(r_ri[:,:3], r_con[:,:3]), np.allclose(r_ri[:,:3], part[:,:3]))
        # print(i, np.linalg.norm(r_ri[:,:3] - r_con[:,:3]), np.linalg.norm(r_ri[:,:3] - part[:,:3]))

        err = np.abs(r_ri[:,3]-r_con[:,3])
        norm = np.dot(r_con[:,3],part[:,3])
        int_err = np.dot(err,part[:,3])*100/norm
        errs[i-1] = int_err
        g.write(str(i)+'    '+str(int_err)+'\n')
        g.flush()

    g.close()
    av_err = np.average(errs)
    sem = np.std(errs)/np.sqrt(ndata)

    print('% MAE =', av_err)
#    print(round(time.time() - start_time,1),'seconds')

main()
