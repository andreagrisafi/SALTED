import numpy as np
import time
import sys
import os
from ase.io import read
import os.path as osp
from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, check_MPI_tasks_count, detect_mpi, distribute_jobs
from salted.cp2k.utils import build_matrices, build_matrices_prim, gto_rec, gto_rec_prim, build_contraction_matrix, get_reciprocal_grid, get_basis_set_info_numba
from numba import types
from numba.typed import Dict
from mpi4py import MPI
from salted.sys_utils import ParseConfig, detect_mpi

inp = ParseConfig().parse_input()

(saltedname, saltedpath, saltedtype,
filename, species, average,
path2qm, qmcode, qmbasis, dfmetric, dfbasis,
filename_pred, predname, predict_data, alpha_only,
rep1, rcut1, sig1, nrad1, nang1, neighspe1,
rep2, rcut2, sig2, nrad2, nang2, neighspe2,
sparsify, nsamples, ncut,
zeta, Menv, Ntrain, trainfrac, regul, eigcut,
gradtol, restart, trainsel, nspe1, nspe2, HP1, HP2) = ParseConfig().get_all_params()

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
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

bdir = osp.join(saltedpath,"basis")

lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, dfbasis, bdir)    

structure = xyzfile[0]
b2a = 0.529177249
atomic_symbols = structure.get_chemical_symbols()
natoms = len(atomic_symbols)
cell = structure.cell/b2a

ntype = {}

for spe in species:
    ntype[spe] = 0
for iat in range(natoms):
    spe = atomic_symbols[iat]
    ntype[spe] += 1
ncoefs = 0
for spe in species:
    ncoefs += nbasis[spe]*ntype[spe]

volume = structure.get_volume()/(b2a**3)

cubefile = open(os.path.join(inp.qm.path2qm, f"conf_{conf_start+1}", inp.qm.cubefile),"r")
lines = cubefile.readlines()
nside = {}
nside[0] = int(lines[3].split()[0])
nside[1] = int(lines[4].split()[0])
nside[2] = int(lines[5].split()[0])
npoints = 1
for i in range(3):
    npoints *= nside[i]
print("Number of grid points:", npoints)
dx = float(lines[3].split()[1])
dy = float(lines[4].split()[2])
dz = float(lines[5].split()[3])
origin = np.asarray(lines[2].split(),dtype=float)[1:4]
rho_qm = []
for line in lines[6+natoms:]:
    rhovals = np.asarray(line.split(),float)
    for rhoval in rhovals:
        rho_qm.append(rhoval)
cubefile.close()
if npoints!=len(rho_qm):
    print("ERROR: inconsistent number of grid points!")
    sys.exit(0)

nx,ny,nz = nside[0], nside[1], nside[2]

Gvec = get_reciprocal_grid(nx,ny,nz,dx,dy,dz)

mask = (
    (Gvec[:, 2] > 0) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0)))

Gvec_half = Gvec[mask]

df_metric = inp.qm.dfmetric

if df_metric == "coulomb":
    Gvec_half = Gvec_half[1:]

nG_half=len(Gvec_half)

# Sort G
knorm_vec = np.linalg.norm(Gvec_half, axis=1)
sort_idx = np.argsort(knorm_vec)
Gvec_half = Gvec_half[sort_idx]
knorm_vec = knorm_vec[sort_idx]

# Compute primitive coefficients
time_a = time.time()
#partial_wave_coefs = gto_rec(lmax_numba, nmax_numba, nbasis, species, npgf, contranorm, alphas, Gvec_half, nG_half) # Contracted
partial_wave_coefs_prim = gto_rec_prim(lmax_numba, species, npgf, alphas, Gvec_half, nG_half) # Primitive
time_b = time.time()
print("Time to compute partial wave coefficients:", time_b-time_a)

# init geometry
for iconf in conf_range:
    structure = xyzfile[iconf]
    atomic_symbols = structure.get_chemical_symbols()
    natoms = len(atomic_symbols)
    cell = structure.cell/b2a
    coords  = structure.positions/b2a

    cube_dir = os.path.join(inp.qm.path2qm, f"conf_{conf_start+iconf+1}", inp.qm.cubefile)

    cubefile = open(cube_dir,"r")
    lines = cubefile.readlines()
    nside = {}
    nside[0] = int(lines[3].split()[0])
    nside[1] = int(lines[4].split()[0])
    nside[2] = int(lines[5].split()[0])
    npoints = 1
    for i in range(3):
        npoints *= nside[i]
    print("Number of grid points:", npoints)
    dx = float(lines[3].split()[1])
    dy = float(lines[4].split()[2])
    dz = float(lines[5].split()[3])
    origin = np.asarray(lines[2].split(),dtype=float)[1:4]
    rho_qm = []
    for line in lines[6+natoms:]:
        rhovals = np.asarray(line.split(),float)
        for rhoval in rhovals:
            rho_qm.append(rhoval)
    cubefile.close()
    if npoints!=len(rho_qm):
        print("ERROR: inconsistent number of grid points!")
        sys.exit(0)

    rho_KS = np.array(rho_qm)

    rho_KS = rho_KS.reshape((nside[0], nside[1], nside[2]))
    
    rho_KS_rec = np.fft.fftn(rho_KS).ravel()* dx * dy * dz
    rho_KS_rec = rho_KS_rec[mask]
    if df_metric == "coulomb":
        rho_KS_rec = rho_KS_rec[1:]
    rho_KS_rec = rho_KS_rec[sort_idx]

    # Build matrices
    time_c = time.time()
    #S, w = build_matrices(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, rho_KS_rec, nG_half, df_metric, rank)
    Sp, wp = build_matrices_prim(Gvec_half, natoms, coords, npgf, lmax_numba, atomic_symbols, partial_wave_coefs_prim, rho_KS_rec, df_metric, rank)

    # Build contraction matrix
    C = build_contraction_matrix(natoms, atomic_symbols, lmax, nmax_numba, npgf, contranorm)

    # Contraction
    S = C.T @ Sp @ C
    w = C.T @ wp
    
    time_d = time.time()
    print("Time to build matrices:", time_d-time_c)

    c = np.linalg.solve(S,w)

    np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{conf_start + iconf}.npy"), c)
    np.save(os.path.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{conf_start + iconf}.npy"), S)
