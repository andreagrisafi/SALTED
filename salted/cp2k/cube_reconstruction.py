import numpy as np
import os
import time
import math
from ase.io import read, write
import os.path as osp
from salted.sys_utils import ParseConfig
from salted.cp2k.utils import init_moments, compute_charge_and_dipole, gto_rec, get_reciprocal_grid, get_basis_set_info_numba
from salted import basis
from numba import njit, prange
from numba import types
from numba.typed import Dict
from numba import get_num_threads, get_thread_id
from mpi4py import MPI


def build(f_list,structure,coefs,cubename,refcube,comm,size,rank):
         
    inp = ParseConfig().parse_input()

    parallel = (size > 1)
    
    print("Numba threads:", get_num_threads())

    [lmax,nmax] = basis.basiset(inp.qm.dfbasis)

    bdir = osp.join(inp.salted.saltedpath,"basis")
    species = inp.system.species
    
    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, inp.qm.dfbasis, bdir)            

    bdir = osp.join(inp.salted.saltedpath,"basis")

    pseudocharge = np.zeros((len(species)), dtype = np.float64)
    pseudocharge_numba = Dict.empty(key_type=types.unicode_type,value_type=types.float64)
    rloc_dict = {}
    for i in range(len(species)):
        spe = species[i]
        pp = np.loadtxt(osp.join(bdir,f"{spe}-local_pseudo.dat"))
        pseudocharge[i] = pp[0]
        pseudocharge_numba[spe] = pp[0] 
        rloc_dict[spe] = pp[1] 

    b2a = 0.529177249
    atomic_symbols = structure.get_chemical_symbols()
    valences = structure.get_atomic_numbers()
    natoms = len(atomic_symbols)
    coords  = structure.positions/b2a

    volume = structure.get_volume()/(b2a**3)

    charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,rank)
    charge, dipole = compute_charge_and_dipole(pseudocharge, natoms, range(natoms), atomic_symbols, coords, lmax, nmax, species, charge_integrals, dipole_integrals, coefs, True, False, False)
    
    if len(refcube)==1:

        # Read reference cube file
        cubefile = open(refcube[0] ,"r")
        lines = cubefile.readlines()
        nside = {}
        nside[0] = int(lines[3].split()[0])
        nside[1] = int(lines[4].split()[0])
        nside[2] = int(lines[5].split()[0])
        npoints = 1
        for i in range(3):
            npoints *= nside[i]
        if rank==0: print("Number of grid points:", npoints)
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

        nx, ny, nz = nside[0], nside[1], nside[2]

    else:
        nx = int(np.floor(structure.cell[0,0]/(0.111*b2a))+1)
        ny = int(np.floor(structure.cell[1,1]/(0.111*b2a))+1)
        nz = int(np.floor(structure.cell[2,2]/(0.111*b2a))+1)

    dx = float(structure.cell[0,0]/nx) / b2a
    dy = float(structure.cell[1,1]/ny) / b2a
    dz = float(structure.cell[2,2]/nz) / b2a

    Gvec = get_reciprocal_grid(nx,ny,nz,dx,dy,dz)

    mask = (
    (Gvec[:, 2] > 0) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0))
    )

    Gvec_half = Gvec[mask][1:]  # Exclude G=0

    nG_half=len(Gvec_half)
    nG = len(Gvec)

    iloc_start = rank*((nG_half//size)+1)
    iloc_end = min(rank*((nG_half//size)+1) + ((nG_half//size)+1), nG_half)
    nG_loc = iloc_end - iloc_start

    #time_pwc = time.time()
    partial_wave_coefs = gto_rec(lmax_numba,nmax_numba,nbasis,species,npgf, contranorm, alphas,Gvec_half[iloc_start:iloc_end], nG_loc)

    #print(time.time()-time_pwc)

    cos_k_coords = np.cos(np.dot(Gvec_half[iloc_start:iloc_end],coords.T))
    sin_k_coords = np.sin(np.dot(Gvec_half[iloc_start:iloc_end],coords.T))

    knorm2_vec = np.sum(Gvec_half[iloc_start:iloc_end]*Gvec_half[iloc_start:iloc_end],axis=1)

    gauss = {}
    for spe in species:
        gauss[spe] = np.exp(-0.5*knorm2_vec*(rloc_dict[spe]**2))

    volfactor = 32.0*np.pi*np.pi/(volume)

    offset = 0
    rho_rec = np.zeros((nG_loc, natoms), dtype=np.complex128)
    if "potential" in f_list or "efield_x" in f_list or "efield_y" in f_list or "efield_z" in f_list or "total_charge" in f_list:
        rho_n_rec = np.zeros((nG_loc, natoms), dtype=np.complex128)

    if "efield_x" in f_list:
        efieldx_rec = np.zeros((nG_loc), dtype=np.complex128)

    if "efield_y" in f_list:
        efieldy_rec = np.zeros((nG_loc), dtype=np.complex128)

    if "efield_z" in f_list:
        efieldz_rec = np.zeros((nG_loc), dtype=np.complex128)

    #time_coefs_dot = time.time()

    for iat in range(natoms):
        spe = atomic_symbols[iat]
        rho_rec[:,iat] = -np.dot(partial_wave_coefs[spe],coefs[offset:offset + nbasis[spe]]) * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat])
        if "potential" in f_list or "efield_x" in f_list or "efield_y" in f_list or "efield_z" in f_list or "total_charge" in f_list:
            rho_n_rec[:,iat] = +pseudocharge_numba[spe] * gauss[spe] * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat]) 
        offset += nbasis[spe]

    #print(time.time()-time_coefs_dot)

    time_energy = time.time()

    if "potential" in f_list:
        ha_rec = ((np.sum(rho_rec + rho_n_rec/(4*np.pi), axis=1))*1/knorm2_vec)

    if "efield_x" in f_list:
        efieldx_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,0]*np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1))

    if "efield_y" in f_list:
        efieldy_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,1]*np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1))

    if "efield_z" in f_list:
        efieldz_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,2]*np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1))

    #print(time.time()-time_energy)

    count = np.ones(size, dtype=int)*((nG_half//size)+1)
    count[size-1] = min(((nG_half//size)+1), nG_half-((size-1)*((nG_half//size)+1)))
    dis = np.arange(0,size,1, dtype=int)*((nG_half//size)+1)

    if "e_density" in f_list:
        rho_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel :
            comm.Allgatherv(np.sum(rho_rec, axis=1), [rho_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            rho_rec_half = np.sum(rho_rec, axis=1)
        rho_rec_half *= 4*np.pi/volume
        rho_rec_global = np.zeros((nG), dtype=np.complex128)

    if "total_charge" in f_list:
        rho_n_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel :
            comm.Allgatherv(np.sum(rho_n_rec, axis=1)/(4*np.pi), [rho_n_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            rho_n_rec_half = np.sum(rho_n_rec, axis=1)/(4*np.pi)
        rho_n_rec_half *= 4*np.pi/volume
        rho_n_rec_global = np.zeros((nG), dtype=np.complex128)
   
    if "potential" in f_list:
        ha_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel:
            comm.Allgatherv(ha_rec, [ha_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            ha_rec_half = ha_rec
        ha_rec_half *= 16*np.pi**2/(volume)
        ha_rec_global = np.zeros((nG), dtype=np.complex128)
    if "efield_x" in f_list:
        efieldx_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel:
            comm.Allgatherv(efieldx_rec, [efieldx_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            efieldx_rec_half = efieldx_rec
        efieldx_rec_half *= 16*np.pi**2/volume
        efieldx_rec_global = np.zeros((nG), dtype=np.complex128)
    if "efield_y" in f_list:
        efieldy_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel:
            comm.Allgatherv(efieldy_rec, [efieldy_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            efieldy_rec_half = efieldy_rec
        efieldy_rec_half *= 16*np.pi**2/volume
        efieldy_rec_global = np.zeros((nG), dtype=np.complex128)
    if "efield_z" in f_list:
        efieldz_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel:
            comm.Allgatherv(efieldz_rec, [efieldz_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            efieldz_rec_half = efieldz_rec
        efieldz_rec_half *= 16*np.pi**2/volume
        efieldz_rec_global = np.zeros((nG), dtype=np.complex128)
   
    idx = np.flatnonzero(mask)[1:]

    # Round Gvec for floating-point safety
    Gvec_rounded = np.round(Gvec, decimals=10)
    Gvec_set = set(tuple(g) for g in Gvec_rounded)

    # For each k in Gvec_half, check if -k exists in Gvec
    valid_k_mask = np.array([
    tuple(np.round(-k, decimals=10)) in Gvec_set
    for k in Gvec_half
    ])

    # Build a dictionary for Gvec (rounded)
    Gvec_tuple = [tuple(g) for g in Gvec_rounded]
    index_map = {g: i for i, g in enumerate(Gvec_tuple)}

    # Find index of -k for each valid k in Gvec_half_valid
    idx_minus_k = np.array([
    index_map[tuple(np.round(-k, decimals=10))]
    for k in Gvec_half[valid_k_mask]
    ])

    if "e_density" in f_list:
        rho_rec_global[idx_minus_k] = np.conj(rho_rec_half[valid_k_mask])
        rho_rec_global[idx] = rho_rec_half
        rho_rec_global[0] = - charge / volume + 0.0j
        rho_rec_global = rho_rec_global.reshape(nx,ny,nz)
        rho = np.real(np.fft.ifftn(rho_rec_global))*nx*ny*nz
        rho = rho.reshape(nx*ny*nz)
    if "total_charge" in f_list:
        rho_n_rec_global[idx_minus_k] = np.conj(rho_n_rec_half[valid_k_mask])
        rho_n_rec_global[idx] = rho_n_rec_half
        rho_n_rec_global[0] = + charge / volume + 0.0j
        rho_n_rec_global = rho_n_rec_global.reshape(nx,ny,nz)
        rho_n = np.real(np.fft.ifftn(rho_n_rec_global))*nx*ny*nz
        rho_n = rho_n.reshape(nx*ny*nz)
    if "potential" in f_list:
        ha_rec_global[idx_minus_k] = np.conj(ha_rec_half[valid_k_mask])
        ha_rec_global[idx] = ha_rec_half
        ha_rec_global[0] = 0.0 + 0.0j
        ha_rec_global = ha_rec_global.reshape(nx,ny,nz)
        ha = np.real(np.fft.ifftn(ha_rec_global))*nx*ny*nz
        ha = ha.reshape(nx*ny*nz)
    if "efield_x" in f_list:
        efieldx_rec_global[idx_minus_k] = np.conj(efieldx_rec_half[valid_k_mask])
        efieldx_rec_global[idx] = efieldx_rec_half
        efieldx_rec_global[0] = 0.0 + 0.0j
        efieldx_rec_global = efieldx_rec_global.reshape(nx,ny,nz)
        efieldx = np.real(np.fft.ifftn(efieldx_rec_global))*nx*ny*nz
        efieldx = efieldx.reshape(nx*ny*nz)
    if "efield_y" in f_list:
        efieldy_rec_global[idx_minus_k] = np.conj(efieldy_rec_half[valid_k_mask])
        efieldy_rec_global[idx] = efieldy_rec_half
        efieldy_rec_global[0] = 0.0 + 0.0j
        efieldy_rec_global = efieldy_rec_global.reshape(nx,ny,nz)
        efieldy = np.real(np.fft.ifftn(efieldy_rec_global))*nx*ny*nz
        efieldy = efieldy.reshape(nx*ny*nz)
    if "efield_z" in f_list:
        efieldz_rec_global[idx_minus_k] = np.conj(efieldz_rec_half[valid_k_mask])
        efieldz_rec_global[idx] = efieldz_rec_half
        efieldz_rec_global[0] = 0.0 + 0.0j
        efieldz_rec_global = efieldz_rec_global.reshape(nx,ny,nz)
        efieldz = np.real(np.fft.ifftn(efieldz_rec_global))*nx*ny*nz
        efieldz = efieldz.reshape(nx*ny*nz)

    if rank==0:

        if "e_density" in f_list:

            # compute integrated electronic charge
            nele = np.sum(rho)*dx*dy*dz
            print("Integral density= ", nele)

            # compute error as a fraction of electronic charge
            if refcube:# and saltedtype=="density":
                error = np.sum(abs(rho+rho_qm))*dx*dy*dz/abs(nele)
                print("% MAE electronic density =", error*100)
        
        dirpath = os.path.join(inp.salted.saltedpath, "cubes")
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        if "e_density" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_e_density.cube","w")
            print("Reconstructed electron density",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(rho[igrid],file=cubef)
            cubef.close()

        if "total_charge" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_total_charge.cube","w")
            print("Reconstructed total charge",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(rho[igrid]+rho_n[igrid],file=cubef)
            cubef.close()

        if "potential" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_potential.cube","w")
            print("Reconstructed hartree potential",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(ha[igrid],file=cubef)
            cubef.close()

        if "efield_x" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_efield_x.cube","w")
            print("Reconstructed electric field along x",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(efieldx[igrid],file=cubef)
            cubef.close()

        if "efield_y" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_efield_y.cube","w")
            print("Reconstructed electric field along y",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(efieldy[igrid],file=cubef)
            cubef.close()

        if "efield_z" in f_list:
            # print density on a cube file
            cubef = open(inp.salted.saltedpath+"cubes/"+cubename + "_efield_z.cube","w")
            print("Reconstructed electric field along z",file=cubef)
            print("CUBE FORMAT",file=cubef)
            print(natoms,file=cubef)
            metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
            print(nx, metric[0,0], metric[0,1], metric[0,2],file=cubef)
            print(ny, metric[1,0], metric[1,1], metric[1,2],file=cubef)
            print(nz, metric[2,0], metric[2,1], metric[2,2],file=cubef)
            for iat in range(natoms):
                print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
            for igrid in range(nx*ny*nz):
                print(efieldz[igrid],file=cubef)
            cubef.close()
   
if __name__ == "__main__":
    build()
