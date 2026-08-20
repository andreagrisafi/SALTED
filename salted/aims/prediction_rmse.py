import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import sparse

from salted.constants import bohr2angs
from salted import basis
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx,
    init_property_file,
    read_system,
)

def build():

    inp = ParseConfig().parse_input()
    # frequently used parameters
    saltedname = inp.salted.saltedname
    saltedpath = inp.salted.saltedpath
    saltedtype = inp.salted.saltedtype
    average = inp.system.average
    qmcode = inp.qm.qmcode
    zeta = inp.gpr.z
    Menv = inp.gpr.Menv
    
    filename_pred = inp.prediction.filename
    predname = inp.prediction.predname
    pdir = inp.prediction.predict_data

    comm, size, rank, parallel = detect_mpi()

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(filename_pred, inp.system.species, inp.qm.dfbasis)
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    testrange = range(ndata)

    # Distribute structures to tasks
    if parallel:
        check_MPI_tasks_count(comm, len(testrange))
        testrange = distribute_jobs(comm, testrange)
        if inp.salted.verbose:
            print(f"Task {rank} handles the following structures: {format_index_ranges(testrange,True)}", flush=True)

    reg_log10_intstr = str(int(np.log10(inp.gpr.regul)))

    if parallel:
        comm.Barrier()

    if average:
        # Load spherical averages 
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))
    
    # Initialize files for validation results
    
    pfname = osp.join(saltedpath,pdir,"errors.dat")
    if rank == 0 and os.path.exists(pfname):
        os.remove(pfname)

    if parallel:
        comm.Barrier()
    
    efile = open(pfname,"a")

    error_density = 0
    variance = 0
    for iconf in testrange:

        overl = np.loadtxt(osp.join(
            saltedpath, pdir, f"{iconf+1}", "ri_ovlp.out"
        ))

        if saltedtype=="density":

            # Load reference coefficients
            ref_coefs = np.loadtxt(osp.join(
                saltedpath, pdir, f"{iconf+1}", "ri_restart_coeffs_df.out"
            ))

            Tsize = len(ref_coefs)
            overl = overl.reshape(Tsize,Tsize)
            ref_projs = np.dot(overl,ref_coefs)

            # load prediction
            pred_coefs = np.loadtxt(osp.join(
                saltedpath, pdir, f"{iconf+1}", "ri_restart_coeffs_ml.out"
            ))

            if average:
                # Compute vector of isotropic average coefficients
                Av_coeffs = np.zeros(Tsize)
                i = 0
                for iat in range(natoms[iconf]):
                    spe = atomic_symbols[iconf][iat]
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                Av_coeffs[i] = av_coefs[spe][n]
                            i += 2*l+1
            
            # Compute predicted density projections <phi|rho>
            pred_projs = np.dot(overl,pred_coefs)

            # compute error
            error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
            error_density += error
            if average:
                ref_projs -= np.dot(overl,Av_coeffs)
                ref_coefs -= Av_coeffs
            var = np.dot(ref_coefs,ref_projs)
            variance += var
            print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
            if inp.salted.verbose:
                print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

        elif saltedtype=="density-response":

            cart = ["x","y","z"]

            ref_coefs = {}
            ref_projs = {}
            pred_coefs = {}
            pred_projs = {}

            error = 0
            var = 0
            j = 1
            for icart in cart:

                # Load reference coefficients for the given Cartesian component
                ref_coefs[icart] = np.loadtxt(osp.join(
                    saltedpath, pdir, f"{iconf+1}", f"ri_rho1_restart_coeffs_{j}_df.out"
                ))
                Tsize = len(ref_coefs[icart])
                overl = overl.reshape(Tsize,Tsize)
                ref_projs[icart] = np.dot(overl,ref_coefs[icart])

                pred_coefs[icart] = np.loadtxt(osp.join(
                    saltedpath, pdir, f"{iconf+1}", f"ri_rho1_restart_coeffs_{j}_ml.out"
                ))

                # Compute predicted density-response projections <phi|rho>
                pred_projs[icart] = np.dot(overl,pred_coefs[icart])

                # Compute error
                error += np.dot(pred_coefs[icart]-ref_coefs[icart],pred_projs[icart]-ref_projs[icart])
                var += np.dot(ref_coefs[icart],ref_projs[icart])
                j += 1

            error_density += error
            variance += var
            print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
            if inp.salted.verbose:
                print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

    efile.close()

    if parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)

    if rank == 0:
        print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", flush=True)


if __name__ == "__main__":
    build()
