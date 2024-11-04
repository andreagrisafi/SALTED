import os
import sys
import time
import os.path as osp

import numpy as np
from scipy import sparse

from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range, init_property_file
from salted.cp2k.utils import init_moments, compute_charge_and_dipole, compute_polarizability

def build():

    inp = ParseConfig().parse_input()
    (saltedname, saltedpath, saltedtype,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    if parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    vdir = f"validations_{saltedname}"
    rdir = f"regrdir_{saltedname}"
    fdir = f"rkhs-vectors_{saltedname}"

    # define test set
    trainrangetot = np.loadtxt(osp.join(
        saltedpath, rdir, f"training_set_N{Ntrain}.txt"
    ), int)
    ntrain = round(trainfrac*len(trainrangetot))
    testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

    # Distribute structures to tasks
    ntest = len(testrange)
    if parallel:
        testrange = get_conf_range(rank,size,ntest,testrange)
        testrange = comm.scatter(testrange,root=0)
        print(f"Task {rank+1} handles the following structures: {testrange}", flush=True)

    reg_log10_intstr = str(int(np.log10(regul)))

    # load regression weights
    weights = np.load(osp.join(
        saltedpath, rdir, f"M{Menv}_zeta{zeta}", f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    dirpath = os.path.join(saltedpath, vdir, f"M{Menv}_zeta{zeta}", f"N{ntrain}_reg{reg_log10_intstr}")
    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if saltedtype=="density-response":
            for icart in ["x","y","z"]:
                cartpath = os.path.join(dirpath, f"{icart}")
                if not os.path.exists(cartpath):
                    os.mkdir(cartpath)
    if size > 1: comm.Barrier()

    if average:
        # Load spherical averages 
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    if qmcode=="cp2k":
        from ase.io import read
        xyzfile = read(filename, ":")
        # Initialize calculation of density/density-response moments
        alphas,sigmas,charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,rank)

    # Initialize files for validation results
    efile = init_property_file("errors",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
    if qmcode=="cp2k": 
        if saltedtype=="density":
            qfile = init_property_file("charges",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
            dfile = init_property_file("dipoles",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
        if saltedtype=="density-response":
            pfile = init_property_file("polarizabilities",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)

    error_density = 0
    variance = 0
    for iconf in testrange:

        overl = np.load(osp.join(
            saltedpath, "overlaps", f"overlap_conf{iconf}.npy"
        ))

        if saltedtype=="density":

            # Load reference coefficients
            ref_coefs = np.load(osp.join(
                saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"
            ))
            ref_projs = np.dot(overl,ref_coefs)
            Tsize = len(ref_coefs)

            # Load RKHS descriptor
            psivec = sparse.load_npz(osp.join(
                saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}.npz"
            ))
            psi = psivec.toarray()

            # Perform prediction
            pred_coefs = np.dot(psi,weights)

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
                # Add spherical averages back 
                pred_coefs += Av_coeffs
            
            # Compute predicted density projections <phi|rho>
            pred_projs = np.dot(overl,pred_coefs)

            np.savetxt(osp.join(dirpath,
                                f"COEFFS-{iconf+1}.dat"
            ), pred_coefs)

            if qmcode=="cp2k":

                # Compute reference total charges and dipole moments
                ref_charge, ref_dipole = compute_charge_and_dipole(xyzfile[iconf],inp.qm.pseudocharge,natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,ref_coefs,average)
                # Compute predicted total charges and dipole moments
                charge, dipole = compute_charge_and_dipole(xyzfile[iconf],inp.qm.pseudocharge,natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs,average)
                

                # Save charges and dipole moments
                print(iconf+1,ref_charge,
                                  charge,file=qfile)
                print(iconf+1,ref_dipole["x"],ref_dipole["y"],ref_dipole["z"],
                                  dipole["x"],    dipole["y"],    dipole["z"],file=dfile)

            
            # compute error
            error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
            error_density += error
            if average:
                ref_projs -= np.dot(overl,Av_coeffs)
                ref_coefs -= Av_coeffs
            var = np.dot(ref_coefs,ref_projs)
            variance += var
            print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
            print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

                
        elif saltedtype=="density-response":

            cart = ["x","y","z"]

            ref_coefs = {}
            ref_projs = {}
            pred_coefs = {}
            pred_projs = {}

            error = 0
            var = 0
            for icart in cart:

                # Load reference coefficients for the given Cartesian component
                ref_coefs[icart] = np.load(osp.join(
                    saltedpath, "coefficients", f"{icart}/coefficients_conf{iconf}.npy"
                ))
                ref_projs[icart] = np.dot(overl,ref_coefs[icart])
                Tsize = len(ref_coefs[icart])

                # Load RKHS descriptor for the given Cartesian component
                psivec = sparse.load_npz(osp.join(
                    saltedpath, fdir, f"M{Menv}_zeta{zeta}", f"psi-nm_conf{iconf}_{icart}.npz"
                ))
                psi = psivec.toarray()

                # Perform prediction
                pred_coefs[icart] = np.dot(psi,weights)

                # Compute predicted density-response projections <phi|rho>
                pred_projs[icart] = np.dot(overl,pred_coefs[icart])

                np.savetxt(osp.join(dirpath,
                                    f"{icart}", 
                                    f"COEFFS-{iconf+1}.dat"
                ), pred_coefs[icart])

                # Compute error
                error += np.dot(pred_coefs[icart]-ref_coefs[icart],pred_projs[icart]-ref_projs[icart])
                var += np.dot(ref_coefs[icart],ref_projs[icart])

            if qmcode=="cp2k":

                # Compute reference and predicted polarizabilities
                ref_alpha = compute_polarizability(xyzfile[iconf],natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,ref_coefs)
                alpha = compute_polarizability(xyzfile[iconf],natoms[iconf],atomic_symbols[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs)
                
                # Save polarizabilities
                print(iconf+1,ref_alpha[("x","x")],ref_alpha[("x","y")],ref_alpha[("x","z")],
                              ref_alpha[("y","x")],ref_alpha[("y","y")],ref_alpha[("y","z")],
                              ref_alpha[("z","x")],ref_alpha[("z","y")],ref_alpha[("z","z")],
                                  alpha[("x","x")],    alpha[("x","y")],    alpha[("x","z")],
                                  alpha[("y","x")],    alpha[("y","y")],    alpha[("y","z")],
                                  alpha[("z","x")],    alpha[("z","y")],    alpha[("z","z")],
                                  file=pfile)
            

            error_density += error
            variance += var
            print(f"{iconf+1:d} {(np.sqrt(error/var)*100):.3e}", file=efile)
            print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)    

    efile.close()
    if qmcode == "cp2k":
        if saltedtype=="density":
            qfile.close()
            dfile.close()
        if saltedtype=="density-response":
            pfile.close()

    if parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)

    if rank == 0:
        print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", flush=True)


if __name__ == "__main__":
    build()
