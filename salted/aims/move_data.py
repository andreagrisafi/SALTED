import os
import os.path as osp

import numpy as np
from ase.io import read
from salted.sys_utils import ParseConfig, detect_mpi, distribute_jobs

def build():
    inp = ParseConfig().parse_input()

    comm, size, rank, parallel = detect_mpi()
    
    if (rank == 0):
        """check if all subdirectories exist, if not create them"""
        sub_dirs = [
            osp.join(inp.salted.saltedpath, d)
            for d in ("overlaps", "coefficients", "projections")
        ]
        if inp.system.collinear:
            sub_dirs = [
            osp.join(inp.salted.saltedpath, d)
            for d in ("overlaps", "coefficients_avgs", "projections_avgs", "coefficients_diff", "projections_diff")
        ]

        for sub_dir in sub_dirs:
            if not osp.exists(sub_dir):
                os.mkdir(sub_dir)
    
    xyzfile = read(inp.system.filename,":")
    ndata = len(xyzfile)

    # Distribute structures to tasks
    if parallel:
        conf_range = distribute_jobs(comm, list(range(ndata)))
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
    
        o = np.loadtxt(osp.join(dirpath, 'ri_projections.out')).reshape(-1)
        t = np.loadtxt(osp.join(dirpath, 'ri_restart_coeffs_df.out')).reshape(-1)
        ovlp = np.loadtxt(osp.join(dirpath, 'ri_ovlp.out')).reshape(-1)

        if inp.system.collinear:
            o_beta = np.loadtxt(osp.join(dirpath, 'ri_projections_beta.out')).reshape(-1)
            t_beta = np.loadtxt(osp.join(dirpath, 'ri_restart_coeffs_beta.out')).reshape(-1)

        n = len(o)
        ovlp = ovlp.reshape(n,n)
        

        if reorder:
            idx = np.loadtxt(osp.join(dirpath, 'idx_prodbas.out')).astype(int)
            cs_list = np.loadtxt(osp.join(dirpath, 'prodbas_condon_shotley_list.out')).astype(int)
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
                
        if inp.system.collinear:
            
            """finding the average"""
            o_avgs = (o + o_beta)*0.5
            t_avgs = (t + t_beta)*0.5

            """finding the difference"""
            o_diff = o - o_beta
            t_diff = t - t_beta
            
            """saves to new location with new name"""
            np.save(osp.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{i}.npy"), ovlp)
            np.save(osp.join(inp.salted.saltedpath, "projections_avgs", f"projections_conf{i}.npy"), o_avgs)
            np.save(osp.join(inp.salted.saltedpath, "coefficients_avgs", f"coefficients_conf{i}.npy"), t_avgs)
            np.save(osp.join(inp.salted.saltedpath, "projections_diff", f"projections_conf{i}.npy"), o_diff)
            np.save(osp.join(inp.salted.saltedpath, "coefficients_diff", f"coefficients_conf{i}.npy"), t_diff)
        else:
            np.save(osp.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{i}.npy"), ovlp)
            np.save(osp.join(inp.salted.saltedpath, "projections", f"projections_conf{i}.npy"), o)
            np.save(osp.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{i}.npy"), t)
    
    if parallel:
        comm.Barrier()
    
    """delte ri basis overlap and proj coeffs files"""
    
    for i in conf_range:
        dirpath = osp.join(inp.qm.path2qm, 'data', str(i+1))
        os.remove(osp.join(dirpath, 'ri_ovlp.out'))
        os.remove(osp.join(dirpath, 'ri_projections.out'))
        if inp.system.collinear:
            os.remove(osp.join(dirpath, 'ri_projections_beta.out'))

if __name__ == "__main__":
    build()
