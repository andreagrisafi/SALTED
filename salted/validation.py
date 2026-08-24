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
    
    if qmcode=='cp2k':
        from salted.cp2k.utils import init_moments, compute_charge_and_dipole, compute_polarizability, compute_hartree_energy, get_basis_set_info_numba, read_local_pseudo, get_reciprocal_grid, build_gcutoff, gto_rec_prim, build_contraction_matrix, get_rho_n

    comm, size, rank, parallel = detect_mpi()

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    vdir = f"validations_{saltedname}"
    rdir = f"regrdir_{saltedname}"
    fdir = f"rkhs-vectors_{saltedname}"

    # define test set
    trainrangetot = np.loadtxt(osp.join(
        saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"
    ), int)
    ntrain = round(inp.gpr.trainfrac*len(trainrangetot))
    testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

    # Distribute structures to tasks
    if parallel:
        check_MPI_tasks_count(comm, len(testrange))
        testrange = distribute_jobs(comm, testrange)
        if inp.salted.verbose:
            print(f"Task {rank} handles the following structures: {format_index_ranges(testrange,True)}", flush=True)

    reg_log10_intstr = str(int(np.log10(inp.gpr.regul)))

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
    if parallel:
        comm.Barrier()

    if average:
        # Load spherical averages 
        av_coefs = {}
        for spe in species:
            av_coefs[spe] = np.load(os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy"))

    if qmcode=="cp2k":
        from ase.io import read
        xyzfile = read(inp.system.filename, ":")
        # Initialize calculation of density/density-response moments
        charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,rank)
        # Basis set and local pseudopotential info
        bdir = osp.join(saltedpath, "basis")
        pseudocharge, rloc = read_local_pseudo(species, bdir)
        if inp.qm.dfmetric == "coulomb":
            lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, inp.qm.dfbasis, bdir)

    # Initialize files for validation results
    efile = init_property_file("errors",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
    if qmcode=="cp2k": 
        if saltedtype=="density":
            qfile = init_property_file("charges",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
            dfile = init_property_file("dipoles",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
            if inp.qm.dfmetric=="coulomb": ufile = init_property_file("electrostatic_energy",saltedpath,vdir,Menv,zeta,ntrain,reg_log10_intstr,rank,size,comm)
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

            if average:
                ref_coefs += Av_coeffs

            if qmcode=="cp2k":

                # Compute reference total charges and dipole moments
                ref_charge, ref_dipole = compute_charge_and_dipole(pseudocharge,natoms[iconf],np.arange(natoms[iconf]),atomic_symbols[iconf],atomic_coords[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,ref_coefs,average,False,comm)
                
                # Compute predicted total charges and dipole moments
                charge, dipole = compute_charge_and_dipole(pseudocharge,natoms[iconf],np.arange(natoms[iconf]),atomic_symbols[iconf],atomic_coords[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs,average,False,comm)

                if inp.qm.dfmetric == "coulomb":
                   
                    # Prepare Hartree energy calculation 
                    structure = xyzfile[iconf]
                    cell = np.asarray(structure.get_cell()) / bohr2angs
                    nx = int(np.floor(cell[0,0]/(0.111))+1)
                    ny = int(np.floor(cell[1,1]/(0.111))+1)
                    nz = int(np.floor(cell[2,2]/(0.111))+1)
                    dx, dy, dz = cell[0,0]/nx, cell[1,1]/ny, cell[2,2]/nz

                    # Generate G-vectors for the half-space Z>0
                    Gvec = get_reciprocal_grid(nx, ny, nz, dx, dy, dz)
                    mask = (
                        (Gvec[:, 2] > 0) |
                        ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
                        ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0)))
                    Gvec_half = Gvec[mask][1:]
                    
                    # Sort array of G-vectors depending on G-norm
                    knorm_vec = np.linalg.norm(Gvec_half, axis=1)
                    sort_idx = np.argsort(knorm_vec)
                    Gvec_half = Gvec_half[sort_idx]
                    knorm_vec = knorm_vec[sort_idx]

                    # Flexible G-cutoffs, primitive pw coefs and contraction matrix
                    gcuts = build_gcutoff(alphas, species, lmax, knorm_vec)
                    pwc_prim_re, pwc_prim_im = gto_rec_prim(lmax_numba, species, npgf, alphas, Gvec_half, gcuts)
                    C = build_contraction_matrix(natoms[iconf], atomic_symbols[iconf], lmax, nmax_numba, npgf, contranorm)

                    # Core charge density in reciprocal space
                    origin = np.zeros(3)
                    rho_n = get_rho_n([nx, ny, nz], dx, dy, dz, origin, natoms[iconf], atomic_coords[iconf], atomic_symbols[iconf], pseudocharge, rloc)
                    rho_n_rec = np.fft.fftn(rho_n).ravel() * dx * dy * dz
                    rho_n_rec = rho_n_rec[mask][1:][sort_idx]

                    # Compute reference Hartree energy
                    ref_hartree, ref_ee, ref_en, ref_nn = compute_hartree_energy(ref_coefs, overl, atomic_coords[iconf], atomic_symbols[iconf], rho_n_rec, Gvec_half, knorm_vec, lmax_numba, npgf, pwc_prim_re, pwc_prim_im, C, gcuts, cell, [nx, ny, nz]) 
                    
                    # Compute predicted Hartree energy
                    hartree, ee, en, nn = compute_hartree_energy(pred_coefs, overl, atomic_coords[iconf], atomic_symbols[iconf], rho_n_rec, Gvec_half, knorm_vec, lmax_numba, npgf, pwc_prim_re, pwc_prim_im, C, gcuts, cell, [nx, ny, nz]) 

                ## Compute reference energy and forces
                #ref_U_ele, ref_forces = elec_energy_forces(lmax,nmax,saltedpath,inp.qm.dfbasis,species,pseudocharge,rloc_dict,structure,ref_coefs)
                #
                ## Compute predicted energy and forces
                #U_ele, forces = elec_energy_forces(lmax,nmax,saltedpath,inp.qm.dfbasis,species,pseudocharge,rloc_dict,structure,pred_coefs)

                # Save total charge 
                print(iconf+1,ref_charge,
                                  charge,file=qfile)

                # Save total dipole
                print(iconf+1,ref_dipole["x"],ref_dipole["y"],ref_dipole["z"],
                                  dipole["x"],    dipole["y"],    dipole["z"],file=dfile)
                
                # Save electrostatic energy
                if inp.qm.dfmetric=="coulomb":
                     print(iconf+1,ref_hartree,
                                       hartree,file=ufile)
            
            np.savetxt(osp.join(dirpath,
                                f"COEFFS-{iconf+1}.dat"
            ), pred_coefs)

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
                ref_alpha = compute_polarizability(natoms[iconf],atomic_symbols[iconf],atomic_coords[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,ref_coefs)
                alpha = compute_polarizability(natoms[iconf],atomic_symbols[iconf],atomic_coords[iconf],lmax,nmax,species,charge_integrals,dipole_integrals,pred_coefs)

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
            if inp.salted.verbose:
                print(f"{iconf+1}: {(np.sqrt(error/var)*100):.3e} % RMSE", flush=True)

    efile.close()
    if qmcode == "cp2k":
        if saltedtype=="density":
            qfile.close()
            dfile.close()
            if inp.qm.dfmetric=="coulomb": ufile.close()
        if saltedtype=="density-response":
            pfile.close()

    if parallel:
        error_density = comm.allreduce(error_density)
        variance = comm.allreduce(variance)

    if rank == 0:
        print(f"\n % RMSE: {(100*np.sqrt(error_density/variance)):.3e}", flush=True)


if __name__ == "__main__":
    build()
