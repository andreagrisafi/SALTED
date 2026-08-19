import numpy as np
import time
import sys
import os
import glob
from ase.io import read
import os.path as osp

from salted.constants import bohr2angs
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, check_MPI_tasks_count, detect_mpi, distribute_jobs
from salted.cp2k.utils import gto_rec, gto_rec_prim, gto_rec_g0, gto_rec_ewald, get_reciprocal_grid, get_basis_set_info_numba, read_local_pseudo, setup_pyscf_species, setup_pyscf_ewald
from salted.cp2k.utils import build_contraction_matrix, get_w_prim, compute_hartree_energy, build_gcutoff, pair_cutoffs, overlap_identity, overlap_coulomb_rec, overlap_coulomb_real
from mpi4py import MPI

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
    rcut_pairs = pair_cutoffs(species, lmax, alphas)
    #for spe1 in species:
    #    for spe2 in species:
    #        print(f'rcut({spe1}-{spe2})={rcut_pairs[(spe1, spe2)]}')
if df_metric == "coulomb":
    sigma_ewald = 2.0 / bohr2angs # 2 angstrom, hard-coded
    pyscf_ewald = setup_pyscf_ewald(species, lmax, nmax_numba, alphas, contranorm, sigma_ewald)
    rcut_pairs = {}
    for spe1 in species:
        for spe2 in species:
            rcut_pairs[(spe1, spe2)] = 4 * (sigma_ewald + sigma_ewald)
            #print(f'rcut({spe1}-{spe2})={rcut_pairs[(spe1, spe2)]}')

for iconf in conf_range:

    if inp.salted.verbose: print("conf:", iconf+1)

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
    with open(glob.glob(cubefile_pattern)[0], "r") as cubefile:
        # Header: 2 comment lines + origin line + 3 lattice lines + atom lines
        header = [cubefile.readline() for _ in range(6 + natoms[iconf])]
        line_ori, line_x, line_y, line_z = (header[i].split() for i in range(2, 6))
        nx = int(line_x[0])
        ny = int(line_y[0])
        nz = int(line_z[0])
        dx = float(line_x[1])
        dy = float(line_y[2])
        dz = float(line_z[3])
        origin = np.asarray(line_ori[1:4], dtype=float)
        npoints = nx * ny * nz
        if inp.salted.verbose: print("Number of grid points:", npoints)
        rho_KS = np.fromstring(cubefile.read(), dtype=np.float64, sep=' ')

    if rho_KS.size != npoints:
        print(f"ERROR: inconsistent number of grid points!")
        sys.exit(0)
        
    rho_KS = rho_KS.reshape((nx, ny, nz))
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
    
    # Sort array of G-vectors depending on G-norm
    knorm_vec = np.linalg.norm(Gvec_half, axis=1)
    sort_idx = np.argsort(knorm_vec)
    Gvec_half = Gvec_half[sort_idx]
    knorm_vec = knorm_vec[sort_idx]
    
    # Get flexible G-cutoffs for full primitive basis function representation
    gcuts = build_gcutoff(alphas, species, lmax, knorm_vec)

    if df_metric == "coulomb":
        # G-vector truncation based on omega
        gmax_ewald = 2.0 * np.pi / sigma_ewald
        gmax_ewald_idx = np.searchsorted(knorm_vec, gmax_ewald).astype(np.int64) # Index of the last G-vector below the cutoff

    # Compute density Fourier-components
    rho_KS_rec = np.fft.fftn(rho_KS).ravel()* dx * dy * dz
    rho_KS_rec = rho_KS_rec[mask]
    if df_metric == "coulomb": # exclude G=0
        rho_KS_rec = rho_KS_rec[1:]
    rho_KS_rec = rho_KS_rec[sort_idx]

    # Compute partial-wave coefs as basis set fourier transform
    time_c = time.time()
    partial_wave_coefs_prim_re, partial_wave_coefs_prim_im = gto_rec_prim(lmax_numba, species, npgf, alphas, Gvec_half, gcuts) # Primitive basis set Fourier transform
    pwc_g0, pwc_spread = gto_rec_g0(natoms[iconf], atomic_symbols[iconf], lmax, nmax_numba, npgf, alphas, contranorm, ncoefs) # Monopole and spread
    if inp.salted.verbose: print("Time to compute primitive pw coeffs:", time.time()-time_c)

    # Compute primitive density projections <phi|O|rho> fully in reciprocal space with flexible G-cutoffs
    time_c = time.time()
    wp = get_w_prim(Gvec_half, natoms[iconf], atomic_coords[iconf], npgf, lmax_numba, atomic_symbols[iconf], partial_wave_coefs_prim_re, partial_wave_coefs_prim_im, volume, rho_KS_rec, df_metric, gcuts, rank)
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
    Sinv = np.linalg.solve(S, np.column_stack([w, q]))
    Sinv_w, Sinv_q = Sinv[:, 0], Sinv[:, 1]
    lagmult = (np.dot(q, Sinv_w) - n_elec) / np.dot(q, Sinv_q) # Lagrange multiplier
    c = Sinv_w - lagmult * Sinv_q # Fitted coefficients

    if inp.salted.verbose: print("Time to solve linear system:", time.time()-time_c)

    # Save data
    np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{conf_start + iconf}.npy"), c)
    np.save(os.path.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{conf_start + iconf}.npy"), S)

    # Hartree energy calculation
    time_c = time.time()
    if df_metric == "coulomb":
        sigma_ewald_en = 1.0 / bohr2angs # 1 angstrom, hard-coded
        e_hartree, e_ee, e_en, e_nn = compute_hartree_energy(c, S, cell, atomic_coords[iconf], atomic_symbols[iconf], species, lmax_numba, lmax_max, nmax_numba, npgf, nbasis, alphas, contranorm, pseudocharge, rloc, sigma_ewald_en, origin, [nx, ny, nz])
        print(f"conf {conf_start+iconf+1}: Hartree energy = {e_hartree:.8f} Ha", flush=True)
        if inp.salted.verbose: print(f"conf {conf_start+iconf+1}: E_ee = {e_ee:.8f} Ha, E_en = {e_en:.8f} Ha, E_nn = {e_nn:.8f} Ha", flush=True)
        if inp.salted.verbose: print('Time to compute Hartree energy:', time.time() - time_c)
    
time_end = time.time()
if inp.salted.verbose: print("Total density fitting time:", time_end-time_start)
