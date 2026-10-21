import os
import os.path as osp
import sys
import time

import numpy as np
from mpi4py import MPI
from numba import get_num_threads

from salted import basis
from salted.constants import bohr2angs, hart2ev
from salted.cp2k.utils import (
    compute_charge_and_dipole,
    get_basis_set_info_numba,
    get_reciprocal_grid,
    gto_rec,
    init_moments,
    read_local_pseudo,
)
from salted.sys_utils import ParseConfig


def build(f_list,structure,coefs,cubename,refcube,comm,size,rank,chunk=32768):
         
    inp = ParseConfig().parse_input()

    parallel = (size > 1)
    
    print("Numba threads:", get_num_threads())

    [lmax,nmax] = basis.basiset(inp.qm.dfbasis)

    bdir = osp.join(inp.salted.saltedpath,"basis")
    species = inp.system.species
    
    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, inp.qm.dfbasis, bdir)            

    bdir = osp.join(inp.salted.saltedpath,"basis")
    pseudocharge, rloc = read_local_pseudo(species, bdir)

    atomic_symbols = structure.get_chemical_symbols()
    valences = structure.get_atomic_numbers()
    natoms = len(atomic_symbols)
    coords  = structure.positions / bohr2angs
    cell = structure.get_cell() / bohr2angs
    volume = structure.get_volume() / (bohr2angs**3)

    charge_integrals,dipole_integrals = init_moments(inp,species,lmax,nmax,rank)
    charge, dipole = compute_charge_and_dipole(pseudocharge, natoms, range(natoms), atomic_symbols, coords, lmax, nmax, species, charge_integrals, dipole_integrals, coefs, True, False, False)
    
    if len(refcube)==1:

        # Read reference cube file (electron density)
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
        rho_qm = np.fromstring("".join(lines[6+natoms:]).replace("-0.", " -0."), dtype=np.float64, sep=' ')
        cubefile.close()
        del lines
        if npoints!=len(rho_qm):
           print("ERROR: inconsistent number of grid points!")
           sys.exit(0)
        if np.any(origin != 0.0):
           print("ERROR: reference cube origin is not zero, the reconstruction assumes it is!")
           sys.exit(0)

        nx, ny, nz = nside[0], nside[1], nside[2]

    else:
        nx = int(np.floor(cell[0,0]/0.111)+1)
        ny = int(np.floor(cell[1,1]/0.111)+1)
        nz = int(np.floor(cell[2,2]/0.111)+1)

    dx = float(cell[0,0]/nx) 
    dy = float(cell[1,1]/ny) 
    dz = float(cell[2,2]/nz) 

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

    knorm2_vec = np.sum(Gvec_half[iloc_start:iloc_end]*Gvec_half[iloc_start:iloc_end],axis=1)

    gauss = {}
    for spe in species:
        gauss[spe] = np.exp(-0.5*knorm2_vec*(rloc[spe]**2))

    rho_rec = np.zeros((nG_loc), dtype=np.complex128)
    if "potential" in f_list or "efield_x" in f_list or "efield_y" in f_list or "efield_z" in f_list or "total_charge" in f_list:
        rho_n_rec = np.zeros((nG_loc), dtype=np.complex128)

    if "efield_x" in f_list:
        efieldx_rec = np.zeros((nG_loc), dtype=np.complex128)

    if "efield_y" in f_list:
        efieldy_rec = np.zeros((nG_loc), dtype=np.complex128)

    if "efield_z" in f_list:
        efieldz_rec = np.zeros((nG_loc), dtype=np.complex128)

    #time_coefs_dot = time.time()

    # loop over chunks of local G-vectors: the partial wave coefficients of all of them, (nG_loc, nbasis) per species,
    # would take tens of GB on large systems
    for g0 in range(0, nG_loc, chunk):
        g1 = min(g0 + chunk, nG_loc)
        Gvec_chunk = Gvec_half[iloc_start+g0:iloc_start+g1]

        partial_wave_coefs = gto_rec(lmax_numba,nmax_numba,nbasis,species,npgf, contranorm, alphas,Gvec_chunk, g1-g0)

        offset = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            cos_k_coords = np.cos(np.dot(Gvec_chunk,coords[iat]))
            sin_k_coords = np.sin(np.dot(Gvec_chunk,coords[iat]))
            rho_rec[g0:g1] += -np.dot(partial_wave_coefs[spe],coefs[offset:offset + nbasis[spe]]) * (cos_k_coords - 1j * sin_k_coords)
            if "potential" in f_list or "efield_x" in f_list or "efield_y" in f_list or "efield_z" in f_list or "total_charge" in f_list:
                rho_n_rec[g0:g1] += pseudocharge[spe] * gauss[spe][g0:g1] * (cos_k_coords - 1j * sin_k_coords)
            offset += nbasis[spe]

    #print(time.time()-time_coefs_dot)

    time_energy = time.time()

    if "potential" in f_list:
        ha_rec = ((rho_rec + rho_n_rec/(4*np.pi))*1/knorm2_vec)

    if "efield_x" in f_list:
        efieldx_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,0]*((rho_n_rec/(4*np.pi))+rho_rec))

    if "efield_y" in f_list:
        efieldy_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,1]*((rho_n_rec/(4*np.pi))+rho_rec))

    if "efield_z" in f_list:
        efieldz_rec = 1/knorm2_vec*(-1j *Gvec_half[iloc_start:iloc_end,2]*((rho_n_rec/(4*np.pi))+rho_rec))

    #print(time.time()-time_energy)

    del Gvec_half, knorm2_vec, gauss

    count = np.ones(size, dtype=int)*((nG_half//size)+1)
    count[size-1] = min(((nG_half//size)+1), nG_half-((size-1)*((nG_half//size)+1)))
    dis = np.arange(0,size,1, dtype=int)*((nG_half//size)+1)

    if "e_density" in f_list or "total_charge" in f_list:
        rho_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel :
            comm.Allgatherv(rho_rec, [rho_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            rho_rec_half = rho_rec
        rho_rec_half *= 4*np.pi/volume
        rho_rec_global = np.zeros((nG), dtype=np.complex128)

    if "total_charge" in f_list:
        rho_n_rec_half = np.zeros((nG_half), dtype=np.complex128)
        if parallel :
            comm.Allgatherv(rho_n_rec/(4*np.pi), [rho_n_rec_half, count, dis, MPI.DOUBLE_COMPLEX])
        else:
            rho_n_rec_half = rho_n_rec/(4*np.pi)
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

    # free the local arrays, only the gathered half arrays are used from here on
    del rho_rec
    if "potential" in f_list or "efield_x" in f_list or "efield_y" in f_list or "efield_z" in f_list or "total_charge" in f_list:
        del rho_n_rec
    if "potential" in f_list:
        del ha_rec
    if "efield_x" in f_list:
        del efieldx_rec
    if "efield_y" in f_list:
        del efieldy_rec
    if "efield_z" in f_list:
        del efieldz_rec

    idx = np.flatnonzero(mask)[1:]

    # Gvec is the C-ordered FFT grid, so -k of the k at grid index (i,j,k) is at (-i mod nx, -j mod ny, -k mod nz)
    ijk = np.unravel_index(idx, (nx,ny,nz)) # Get the (i,j,k) indices of the valid k points in Gvec_half
    valid_k_mask = np.ones(len(idx), dtype=bool) # Mask to identify valid k points
    for n, i in zip((nx,ny,nz), ijk):
        if n % 2 == 0: # If the grid size is even
            valid_k_mask &= (i != n//2) # Mask out the Nyquist frequency points for each dimension
    idx_minus_k = np.ravel_multi_index(tuple((-i[valid_k_mask]) % n for n, i in zip((nx,ny,nz), ijk)), (nx,ny,nz)) # Find index of -k for each valid k in Gvec_half

    if "e_density" in f_list or "total_charge" in f_list:
        rho_rec_global[idx_minus_k] = np.conj(rho_rec_half[valid_k_mask])
        rho_rec_global[idx] = rho_rec_half
        rho_rec_global[0] = - charge / volume + 0.0j
        rho_rec_global = rho_rec_global.reshape(nx,ny,nz)
        rho = np.real(np.fft.ifftn(rho_rec_global))*nx*ny*nz
        rho = rho.reshape(nx*ny*nz)
        del rho_rec_global, rho_rec_half
    if "total_charge" in f_list:
        rho_n_rec_global[idx_minus_k] = np.conj(rho_n_rec_half[valid_k_mask])
        rho_n_rec_global[idx] = rho_n_rec_half
        rho_n_rec_global[0] = + charge / volume + 0.0j
        rho_n_rec_global = rho_n_rec_global.reshape(nx,ny,nz)
        rho_n = np.real(np.fft.ifftn(rho_n_rec_global))*nx*ny*nz
        rho_n = rho_n.reshape(nx*ny*nz)
        del rho_n_rec_global, rho_n_rec_half
    if "potential" in f_list:
        ha_rec_global[idx_minus_k] = np.conj(ha_rec_half[valid_k_mask])
        ha_rec_global[idx] = ha_rec_half
        ha_rec_global[0] = 0.0 + 0.0j
        ha_rec_global = ha_rec_global.reshape(nx,ny,nz)
        ha = np.real(np.fft.ifftn(ha_rec_global))*nx*ny*nz
        ha = ha.reshape(nx*ny*nz)
        del ha_rec_global, ha_rec_half
    if "efield_x" in f_list:
        efieldx_rec_global[idx_minus_k] = np.conj(efieldx_rec_half[valid_k_mask])
        efieldx_rec_global[idx] = efieldx_rec_half
        efieldx_rec_global[0] = 0.0 + 0.0j
        efieldx_rec_global = efieldx_rec_global.reshape(nx,ny,nz)
        efieldx = np.real(np.fft.ifftn(efieldx_rec_global))*nx*ny*nz
        efieldx = efieldx.reshape(nx*ny*nz)
        del efieldx_rec_global, efieldx_rec_half
    if "efield_y" in f_list:
        efieldy_rec_global[idx_minus_k] = np.conj(efieldy_rec_half[valid_k_mask])
        efieldy_rec_global[idx] = efieldy_rec_half
        efieldy_rec_global[0] = 0.0 + 0.0j
        efieldy_rec_global = efieldy_rec_global.reshape(nx,ny,nz)
        efieldy = np.real(np.fft.ifftn(efieldy_rec_global))*nx*ny*nz
        efieldy = efieldy.reshape(nx*ny*nz)
        del efieldy_rec_global, efieldy_rec_half
    if "efield_z" in f_list:
        efieldz_rec_global[idx_minus_k] = np.conj(efieldz_rec_half[valid_k_mask])
        efieldz_rec_global[idx] = efieldz_rec_half
        efieldz_rec_global[0] = 0.0 + 0.0j
        efieldz_rec_global = efieldz_rec_global.reshape(nx,ny,nz)
        efieldz = np.real(np.fft.ifftn(efieldz_rec_global))*nx*ny*nz
        efieldz = efieldz.reshape(nx*ny*nz)
        del efieldz_rec_global, efieldz_rec_half

    if rank==0:

        if "e_density" in f_list:

            # compute integrated electronic charge
            nele = np.sum(rho)*dx*dy*dz
            print("Integral density= ", nele)

            # compute error as a fraction of electronic charge
            if len(refcube)==1 and inp.salted.saltedtype=="density":
                error = np.sum(abs(rho+rho_qm))*dx*dy*dz/abs(nele)
                print("% MAE electronic density =", error*100)

            # compute errors of Hartree potential and electric field with respect to the reference density
            if len(refcube)==1 and inp.salted.saltedtype=="density" and any(f in f_list for f in ["potential","efield_x","efield_y","efield_z"]):
                drho_rec = np.fft.fftn((rho+rho_qm).reshape(nx,ny,nz))*dx*dy*dz
                knorm2_global = np.sum(Gvec*Gvec,axis=1).reshape(nx,ny,nz)
                knorm2_global[0,0,0] = 1.0
                dha_rec = 4*np.pi*drho_rec/knorm2_global
                dha_rec[0,0,0] = 0.0 + 0.0j
                del drho_rec, knorm2_global
                if "potential" in f_list:
                    dha = np.real(np.fft.ifftn(dha_rec)).reshape(nx*ny*nz)/(dx*dy*dz)
                    error = np.sum(abs(dha))/np.sum(abs(ha-dha))
                    print("% MAE hartree potential =", error*100, "  ( MAE =", np.mean(abs(dha))*hart2ev*1000, "mV )")
                if "efield_x" in f_list:
                    defieldx = np.real(np.fft.ifftn(-1j*Gvec[:,0].reshape(nx,ny,nz)*dha_rec)).reshape(nx*ny*nz)/(dx*dy*dz)
                    error = np.sum(abs(defieldx))/np.sum(abs(efieldx-defieldx))
                    print("% MAE electric field along x =", error*100, "  ( MAE =", np.mean(abs(defieldx))*hart2ev*1000/bohr2angs, "mV/A )")
                if "efield_y" in f_list:
                    defieldy = np.real(np.fft.ifftn(-1j*Gvec[:,1].reshape(nx,ny,nz)*dha_rec)).reshape(nx*ny*nz)/(dx*dy*dz)
                    error = np.sum(abs(defieldy))/np.sum(abs(efieldy-defieldy))
                    print("% MAE electric field along y =", error*100, "  ( MAE =", np.mean(abs(defieldy))*hart2ev*1000/bohr2angs, "mV/A )")
                if "efield_z" in f_list:
                    defieldz = np.real(np.fft.ifftn(-1j*Gvec[:,2].reshape(nx,ny,nz)*dha_rec)).reshape(nx*ny*nz)/(dx*dy*dz)
                    error = np.sum(abs(defieldz))/np.sum(abs(efieldz-defieldz))
                    print("% MAE electric field along z =", error*100, "  ( MAE =", np.mean(abs(defieldz))*hart2ev*1000/bohr2angs, "mV/A )")
        
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
