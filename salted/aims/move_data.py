import os
import os.path as osp

import numpy as np
from ase.io import read
from salted.sys_utils import ParseConfig, get_conf_range

def build():
    inp = ParseConfig().parse_input()

    if inp.system.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
        print('This is task',rank+1,'of',size,flush=True)
    else:
        rank = 0
        size = 1
    
    if (rank == 0):
        """check if all subdirectories exist, if not create them"""
        sub_dirs = [
            osp.join(inp.salted.saltedpath, d)
            for d in ("overlaps", "coefficients", "projections")
        ]
        for sub_dir in sub_dirs:
            if not osp.exists(sub_dir):
                os.makedirs(sub_dir)

        if inp.salted.saltedtype == 'density-response':
            sub_dirs = [
                osp.join(inp.salted.saltedpath, d, x)
                for d in ("coefficients", "projections") for x in ("x","y","z")
            ]
            for sub_dir in sub_dirs:
                if not osp.exists(sub_dir):
                    os.makedirs(sub_dir)
    
    xyzfile = read(inp.system.filename,":")
    ndata = len(xyzfile)
    
    # Distribute structures to tasks
    if inp.system.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
    else:
        conf_range = list(range(ndata))
    
    def get_reorder_bool(dirpath):
        """Determine the version of FHI-aims used.
        If a version newer than 240403, coefficients are 
        internally reordered on input/output, and the
        SALTED helper functions should not also reorder coefficients.
    
        Args:
            dirpath (string): directory containing AIMS outputs
        Returns:
            boolean: whether SALTED helper functions should reorder
        """
    
        with open(osp.join(dirpath,'aims.out'),'r') as afile:
            for i,line in enumerate(afile):
                if i == 51:
                    if line.split()[:2] == ['FHI-aims','version']:
                        if int(line.split()[-1]) >= 240403:
                            reorder = False
                        else:
                            reorder = True
                        return reorder
                    else:
                        print('The aims.out file does not have the FHI-aims version listed on line 52 as expected')
                    break
                elif i > 51:
                    print('The aims.out file does not have the FHI-aims version listed on line 52 as expected')
                    break
            else:
                print('The aims.out is very short; FHI-aims has not executed properly')
    
    for i in conf_range:
    
        dirpath = osp.join(inp.qm.path2qm, 'data', str(i+1))
        reorder = get_reorder_bool(dirpath)
    
        o = np.fromfile(osp.join(dirpath, 'ri_projections.out'),sep=" ").reshape(-1)
        t = np.fromfile(osp.join(dirpath, 'ri_restart_coeffs_df.out'),sep=" ").reshape(-1)
        ovlp = np.fromfile(osp.join(dirpath, 'ri_ovlp.out'),sep=" ").reshape(-1)
        
        n = len(o)
        ovlp = ovlp.reshape(n,n)
        
        if reorder:
            idx = np.fromfile(osp.join(dirpath, 'idx_prodbas.out'),sep=" ").astype(int)
            cs_list = np.fromfile(osp.join(dirpath, 'prodbas_condon_shotley_list.out'),sep=" ").astype(int)
            idx -= 1
            cs_list -= 1
            idx = list(idx)
            cs_list = list(cs_list)
        
        
            for j in cs_list:
                ovlp[j,:] *= -1
                ovlp[:,j] *= -1
                o[j] *= -1
                t[j] *= -1
        
            o = o[idx]
            t = t[idx]
            ovlp = ovlp[idx,:]
            ovlp = ovlp[:,idx]
        
        np.save(osp.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{i}.npy"), ovlp)
        np.save(osp.join(inp.salted.saltedpath, "projections", f"projections_conf{i}.npy"), o)
        np.save(osp.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{i}.npy"), t)

        """move denisty response projections and coeffs if requested"""

        if inp.salted.saltedtype == 'density-response':
            o = np.fromfile(osp.join(dirpath, 'ri_projections_rho1.out'),sep=" ")
            t = []
            t.append(np.fromfile(osp.join(dirpath, 'ri_rho1_restart_coeffs_1.out'),sep=" ").reshape(-1))
            t.append(np.fromfile(osp.join(dirpath, 'ri_rho1_restart_coeffs_2.out'),sep=" ").reshape(-1))
            t.append(np.fromfile(osp.join(dirpath, 'ri_rho1_restart_coeffs_3.out'),sep=" ").reshape(-1))
            t = np.array(t)
        
            if reorder:
                for j in cs_list:
                    o[:,j] *= -1
                    t[:,j] *= -1
        
                o = o[:,idx]
                t = t[:,idx]

            for j,x in enumerate(['x','y','z']):        
                np.save(osp.join(inp.salted.saltedpath, "projections", x, f"projections_conf{i}.npy"), o[j])
                np.save(osp.join(inp.salted.saltedpath, "coefficients", x, f"coefficients_conf{i}.npy"), t[j])

    
    if size > 1: comm.Barrier()
    
    """delte ri basis overlap and proj coeffs files"""
    
    for i in conf_range:
        dirpath = osp.join(inp.qm.path2qm, 'data', str(i+1))
        os.remove(osp.join(dirpath, 'ri_ovlp.out'))
        os.remove(osp.join(dirpath, 'ri_projections.out'))

        if inp.salted.saltedtype == 'density-response':
            os.remove(osp.join(dirpath, 'ri_projections_rho1.out'))

if __name__ == "__main__":
    build()
