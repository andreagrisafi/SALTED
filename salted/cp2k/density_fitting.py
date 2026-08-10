import numpy as np
import time
import sys
import os
import glob
from ase.io import read
import os.path as osp
from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, check_MPI_tasks_count, detect_mpi, distribute_jobs
from salted.cp2k.utils import gto_rec, gto_rec_prim, gto_rec_g0, gto_rec_ewald, get_reciprocal_grid, get_basis_set_info_numba, read_local_pseudo, setup_pyscf_species, setup_pyscf_ewald
from salted.cp2k.utils import build_contraction_matrix, get_w_prim, compute_hartree_energy, build_gcutoff, pair_cutoffs, build_matrices, overlap_identity, overlap_coulomb_rec, overlap_coulomb_real
from numba import types
from numba.typed import Dict
from mpi4py import MPI
from salted.sys_utils import ParseConfig, detect_mpi

b2a = 0.529177249

inp = ParseConfig().parse_input()
df_metric = inp.qm.dfmetric

conf_start = int(sys.argv[1])
conf_end = int(sys.argv[2])

comm, size, rank, parallel = detect_mpi()

if rank==0: print("Parallel run over", size, "tasks",flush=True)

if rank==0:

    dirpath = os.path.join(inp.salted.saltedpath, "coefficients")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    dirpath = os.path.join(inp.salted.saltedpath, "overlaps")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

xyzfile = read(inp.system.filename,":")[conf_start:conf_end+1]
ndata = len(xyzfile)

if parallel:

    comm.Barrier()

    check_MPI_tasks_count(comm, ndata, "configurations")
    conf_range = distribute_jobs(comm, np.arange(ndata,dtype=int))
    print(
        f"Task {rank+1} handles the following configurations: {conf_range}", flush=True
    )

else:

    conf_range = np.arange(ndata,dtype=int)

# Initialize SALTED
time_start = time.time()
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

bdir = osp.join(inp.salted.saltedpath,"basis")
lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, inp.qm.dfbasis, bdir)    

# PySCF setup and cutoffs
pyscf_data = setup_pyscf_species(species, lmax, nmax_numba, alphas, contranorm)
if df_metric =="identity":
    rcut_pairs = pair_cutoffs(species, lmax, alphas, contranorm, eps=1e-10)
    #for spe1 in species:
    #    for spe2 in species:
    #        print(f'rcut({spe1}-{spe2})={rcut_pairs[(spe1, spe2)]}')
if df_metric == "coulomb":
    sigma_ewald = 2.0 / b2a # 2 angstrom, hard-coded
    pyscf_ewald = setup_pyscf_ewald(species, lmax, nmax_numba, alphas, contranorm, sigma_ewald)
    rcut_pairs = {}
    for spe1 in species:
        for spe2 in species:
            rcut_pairs[(spe1, spe2)] = 4 * (sigma_ewald + sigma_ewald)
            #print(f'rcut({spe1}-{spe2})={rcut_pairs[(spe1, spe2)]}')
            #rcut_pairs[(spe1, spe2)] = 4 * np.sqrt(2 + max(lmax[spe1], lmax[spe2])) * sigma_ewald # More strict criterion, if needed
            #print(f'rcut({spe1}-{spe2})={rcut_pairs[(spe1, spe2)]}')

# init geometry
for iconf in conf_range:

    if inp.salted.verbose: print("conf:", iconf+1)
 
    structure = xyzfile[iconf]

    # Compute coefs array size and number of electrons
    pseudocharge, rloc = read_local_pseudo(species, bdir)
    ntype = {}
    n_elec = 0.0
    for spe in species:
        ntype[spe] = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        ntype[spe] += 1
        n_elec += pseudocharge[spe]
    ncoefs = 0
    for spe in species:
        ncoefs += nbasis[spe]*ntype[spe]

    # Read in electron density from cube files
    cubefile_pattern = os.path.join(inp.qm.path2qm, f"conf_{conf_start+iconf+1}", "*ELECTRON_DENSITY-1_0.cube")
    cubefile = open(glob.glob(cubefile_pattern)[0], "r")
    lines = cubefile.readlines()
    nside = {}
    nside[0] = int(lines[3].split()[0])
    nside[1] = int(lines[4].split()[0])
    nside[2] = int(lines[5].split()[0])
    npoints = 1
    for i in range(3):
        npoints *= nside[i]
    if inp.salted.verbose: print("Number of grid points:", npoints)
    dx = float(lines[3].split()[1])
    dy = float(lines[4].split()[2])
    dz = float(lines[5].split()[3])
    origin = np.asarray(lines[2].split(),dtype=float)[1:4]
    rho_qm = []
    for line in lines[6+natoms[iconf]:]:
        rhovals = np.asarray(line.split(),float)
        for rhoval in rhovals:
            rho_qm.append(rhoval)
    cubefile.close()
    if npoints!=len(rho_qm):
        print("ERROR: inconsistent number of grid points!")
        sys.exit(0)
        
    nx,ny,nz = nside[0], nside[1], nside[2]
    volume = (nx*dx)*(ny*dy)*(nz*dz) 

    # Get cell
    cell = np.zeros((3,3))
    cell[0,0] = nx*dx
    cell[1,1] = ny*dy
    cell[2,2] = nz*dz

    # Generate G-vectors for the half-space Z>0
    Gvec = get_reciprocal_grid(nx,ny,nz,dx,dy,dz)
    mask = (
        (Gvec[:, 2] > 0) |
        ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
        ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0)))
    Gvec_half = Gvec[mask]
    if df_metric == "coulomb": # exclude G=0
        Gvec_half = Gvec_half[1:]
    nG_half = len(Gvec_half)
    
    # Sort array of G-vectors depending on G-norm
    knorm_vec = np.linalg.norm(Gvec_half, axis=1)
    sort_idx = np.argsort(knorm_vec)
    Gvec_half = Gvec_half[sort_idx]
    knorm_vec = knorm_vec[sort_idx]
    
    # Get flexible G-cutoffs for full primitive basis function representation
    gcuts = build_gcutoff(alphas, npgf, species, lmax, knorm_vec, nG_half)

    if df_metric == "coulomb":
        # G-vector truncation based on omega
        gmax_ewald = 2.0 * np.pi / sigma_ewald
        gmax_ewald_idx = np.searchsorted(knorm_vec, gmax_ewald).astype(np.int64) # Index of the last G-vector below the cutoff

    # Compute density Fourier-components
    rho_KS = np.array(rho_qm)
    rho_KS = rho_KS.reshape((nside[0], nside[1], nside[2]))
    rho_KS_rec = np.fft.fftn(rho_KS).ravel()* dx * dy * dz
    rho_KS_rec = rho_KS_rec[mask]
    if df_metric == "coulomb": # exclude G=0
        rho_KS_rec = rho_KS_rec[1:]
    rho_KS_rec = rho_KS_rec[sort_idx]

    # Compute partial-wave coefs as basis set fourier transform
    time_c = time.time()
    partial_wave_coefs_prim = gto_rec_prim(lmax_numba, species, npgf, alphas, Gvec_half, gcuts) # Primitive
    if inp.salted.verbose: print("Time to compute primitive pw coeffs:", time.time()-time_c)

    # Compute primitive density projections <phi|O|rho> fully in reciprocal space with flexible G-cutoffs
    time_c = time.time()
    wp = get_w_prim(Gvec_half, natoms[iconf], atomic_coords[iconf], npgf, lmax_numba, atomic_symbols[iconf], partial_wave_coefs_prim, volume, rho_KS_rec, df_metric, gcuts, rank)
    # Contract projections
    C = build_contraction_matrix(natoms[iconf], atomic_symbols[iconf], lmax, nmax_numba, npgf, contranorm) 
    w = C.T @ wp
    if inp.salted.verbose: print("Time to build w:", time.time()-time_c)
   
    if df_metric == "identity":

        # Compute overlap matrix S_ij = <Phi_i|Phi_j> fully in real space using PySCF int2c routines 
        time_c = time.time()
        S = overlap_identity(cell, atomic_coords[iconf], atomic_symbols[iconf], nbasis, ncoefs, volume, pyscf_data, rcut_pairs)
        if inp.salted.verbose: print("Time to build S:", time.time()-time_c)
        
    elif df_metric == "coulomb":

        # Compute 2-center Coulomb integral matrix J_ij = <Phi_i|1/|r-r'||Phi_j> via Ewald sums 
        time_c = time.time()
        # Short-range term in real space with PySCF
        pwc_g0, pwc_spread = gto_rec_g0(natoms[iconf], atomic_symbols[iconf], lmax, nmax_numba, npgf, alphas, contranorm, ncoefs) # Needed for the G=0 correction
        S_SR = overlap_coulomb_real(cell, atomic_coords[iconf], atomic_symbols[iconf], nbasis, ncoefs, volume, pyscf_data, pyscf_ewald, pwc_g0, pwc_spread, sigma_ewald, rcut_pairs)
        if inp.salted.verbose: print("Time to build S_SR:", time.time()-time_c)
        time_c = time.time()
        # Long-range term in reciprocal space
        partial_wave_coefs = gto_rec(lmax_numba, nmax_numba, nbasis, species, npgf, contranorm, alphas, Gvec_half, gmax_ewald_idx)
        partial_wave_coefs_ewald = gto_rec_ewald(lmax_numba, lmax_max, nmax_numba, nbasis, species, npgf, contranorm, alphas, Gvec_half, gmax_ewald_idx, sigma_ewald)
        if inp.salted.verbose: print("Time to compute G-truncated contracted pw coeffs:", time.time()-time_c)
        time_c = time.time()
        S_LR = overlap_coulomb_rec(Gvec_half, natoms[iconf], atomic_coords[iconf], nbasis, ncoefs, atomic_symbols[iconf], partial_wave_coefs, partial_wave_coefs_ewald, volume, gmax_ewald_idx, rank) 
        if inp.salted.verbose: print("Time to build S_LR:", time.time()-time_c)
        # Collect SR and LR terms 
        S = S_SR + S_LR 

    # Solve density-fitting linear system 
    time_c = time.time()
    q = 4.0 * np.pi * pwc_g0
    Sinv_w = np.linalg.solve(S, w)
    Sinv_q = np.linalg.solve(S, q)
    lagmult = (np.dot(q, Sinv_w) - n_elec) / np.dot(q, Sinv_q) # Lagrange multiplier
    c = Sinv_w - lagmult * Sinv_q

    if inp.salted.verbose: print("Time to solve linear system:", time.time()-time_c)

    # Save data
    np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{conf_start + iconf}.npy"), c)
    np.save(os.path.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{conf_start + iconf}.npy"), S)

    # Hartree energy calculation
    time_c = time.time()
    if df_metric == "coulomb":
        sigma_ewald_en = 1.0 / b2a
        e_hartree, e_ee, e_en, e_nn = compute_hartree_energy(c, S, cell, atomic_coords[iconf], atomic_symbols[iconf], species, lmax_numba, lmax_max, nmax_numba, npgf,
                                                             nbasis, alphas, contranorm, pseudocharge, rloc, sigma_ewald_en, origin, nside)
        print(f"conf {conf_start+iconf+1}: Hartree energy = {e_hartree:.8f} Ha", flush=True)
        if inp.salted.verbose: print(f"conf {conf_start+iconf+1}: E_ee = {e_ee:.8f} Ha, E_en = {e_en:.8f} Ha, E_nn = {e_nn:.8f} Ha", flush=True)
        if inp.salted.verbose: print('Time to compute Hartree energy:', time.time() - time_c)
    
time_end = time.time()
if inp.salted.verbose: print("Total density fitting time:", time_end-time_start)