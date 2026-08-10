import os
import sys
import time
import math
import os.path as osp
import numpy as np
from scipy import special
from numba import njit, prange
from numba import types
from numba.typed import Dict
from pyscf import gto as _pyscf_gto

def init_moments(inp,species,lmax,nmax,rank):
    """Compute basis function integrals relevant for computing total charge, dipole and polarizability tensor"""

    if rank==0:

        if inp.salted.saltedtype=="density":

            print("Total charges and polarization vectors are computed as the zero and first moment of the electron density.")
            print("WARNING: Computed values of polarization vectors have physical meaning only along those Cartesian directions for which the electron density vanishes before reaching the cell periodic boundaries; the modern theory of polarization should be used otherwise.")

        elif inp.salted.saltedtype=="density-response":
            
            print("Polarizability tensors are computed as the first moment of the electron-density electric-field response.")
            print("WARNING: Computed values of polarizability tensors have physical meaning only along those Cartesian directions for which the electron density vanishes before reaching the cell periodic boundaries; the modern theory of polarization should be used otherwise.")

    # Get CP2K basis set information 
    bdir = osp.join(inp.salted.saltedpath,"basis")
    alphas = {}
    contra = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            alphas[(spe,l)] = np.atleast_1d(np.loadtxt(osp.join(bdir,f"{spe}-{inp.qm.dfbasis}-alphas-L{l}.dat")))
            contra[(spe,l)] = np.atleast_2d(np.loadtxt(osp.join(bdir,f"{spe}-{inp.qm.dfbasis}-contra-L{l}.dat")))

    # Compute basis function integrals 
    charge_integrals = {}
    dipole_integrals = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                npgf = len(alphas[(spe,l)])
                # Compute inner product between contracted Gaussian-type functions 
                inner = 0.0
                for ipgf1 in range(npgf):
                    for ipgf2 in range(npgf):
                        # Compute primitive integral \int_0^\infty dr r^2 r^{2l} \exp[-r^2/\sigma^2]
                        inner += contra[(spe,l)][n,ipgf1] * contra[(spe,l)][n,ipgf2] * 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l)][ipgf1] + alphas[(spe,l)][ipgf2])**(l+1.5) )
                # Compute \int_0^\infty dr r^2 r^{2l} \exp[-r^2/\sigma^2]
                #inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                charge_radint = 0.0
                dipole_radint = 0.0
                # Perform contraction over primitive GTOs
                for ipgf in range(npgf):
                    # Compute primitive integral \int_0^\infty dr r^2 r^l \exp[-r^2/(2\sigma^2)]
                    charge_radint += contra[(spe,l)][n,ipgf] * 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l)][ipgf])**(float(l+3)/2.0) )
                    # Compute primitive integral \int_0^\infty dr r^3 r^l \exp[-r^2/(2\sigma^2)]
                    sigma = np.sqrt(0.5/alphas[(spe,l)][ipgf])
                    dipole_radint += contra[(spe,l)][n,ipgf] * 2**float(1.0+float(l)/2.0) * sigma**(4+l) * special.gamma(2.0+float(l)/2.0)
                # Muliply by radial and spherical harmonics normalization factor
                charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
                dipole_integrals[(spe,l,n)] = dipole_radint * np.sqrt(4.0*np.pi/3.0) / np.sqrt(inner)

    return [charge_integrals,dipole_integrals]

def compute_hartree_energy(coefs, S, cell, coords, atomic_symbols, species, lmax_numba, lmax_max, nmax_numba, npgf, nbasis, alphas, contranorm, pseudocharge, rloc, sigma_ewald, origin, nside):

    natoms = len(atomic_symbols)
    ncoefs = len(coefs)

    # Real-space grid
    nx, ny, nz = nside[0], nside[1], nside[2]
    dx, dy, dz = cell[0,0]/nx, cell[1,1]/ny, cell[2,2]/nz
    volume = (nx*dx) * (ny*dy) * (nz*dz)

    # G-vectors for the half-space, G=0 excluded, sorted by |G|
    Gvec = get_reciprocal_grid(nx, ny, nz, dx, dy, dz)
    mask = (
        (Gvec[:, 2] > 0) |
        ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
        ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0)))
    Gvec_half = Gvec[mask][1:]
    knorm_vec = np.linalg.norm(Gvec_half, axis=1)
    sort_idx = np.argsort(knorm_vec)
    Gvec_half = Gvec_half[sort_idx]
    knorm_vec = knorm_vec[sort_idx]

    # Reciprocal space cutoff
    gmax_ewald = 2.0 * np.pi / sigma_ewald
    gmax_ewald_idx = np.searchsorted(knorm_vec, gmax_ewald).astype(np.int64)  # Index of the last G-vector below the cutoff  

    # Real space cutoff
    rcut_pairs = {}
    for spe1 in species:
        for spe2 in species:
            rcut_pairs[(spe1, spe2)] = 4 * (sigma_ewald + sigma_ewald)

    # PySCF objects: Ewald-corrected basis (bra) and core pseudocharge (ket)
    pyscf_ewald = setup_pyscf_ewald(species, lmax_numba, nmax_numba, alphas, contranorm, sigma_ewald)
    pyscf_core = setup_pyscf_core(species, pseudocharge, rloc)

    # Compute partial-wave coefs as basis set fourier transform
    partial_wave_coefs_ewald = gto_rec_ewald(lmax_numba, lmax_max, nmax_numba, nbasis, species, npgf, contranorm, alphas, Gvec_half, gmax_ewald_idx, sigma_ewald)
    pwc_g0, pwc_spread = gto_rec_g0(natoms, atomic_symbols, lmax_numba, nmax_numba, npgf, alphas, contranorm, ncoefs)

    # Core charge density in reciprocal space
    rho_n = get_rho_n(nside, dx, dy, dz, origin, natoms, coords, atomic_symbols, pseudocharge, rloc)
    rho_n_rec = np.fft.fftn(rho_n).ravel() * dx * dy * dz
    rho_n_rec = rho_n_rec[mask][1:][sort_idx]

    # Electron-electron term: E_ee = 1/2 c^T.S.c
    e_ee = 0.5 * np.dot(coefs, S @ coefs)

    # Electron-nucleus term: E_en = -sum_i c_i (Phi_i|rho_n)
    wn = get_wn_real(cell, coords, atomic_symbols, nbasis, ncoefs, volume, pyscf_ewald, pyscf_core, rcut_pairs)
    ztot = sum(pseudocharge[spe] for spe in atomic_symbols)
    wn -= (4.0*np.pi)**2 / (2.0*volume) * ztot * (sigma_ewald**2 * pwc_g0 - pwc_spread)  # G=0 correction
    wn += get_wn_rec(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs_ewald, rho_n_rec, volume, gmax_ewald_idx)
    e_en = -np.dot(coefs, wn)

    # Nucleus-nucleus term: E_nn = 1/2 (rho_n|rho_n)
    e_nn = 0.5 * overlap_coulomb_rho(rho_n_rec, rho_n_rec, knorm_vec, volume)

    return e_ee + e_en + e_nn, e_ee, e_en, e_nn

#def compute_charge_and_dipole(geom,pseudocharge,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,dipole_integrals,coefs,average):
#    """Compute total charge and dipole moment for the given configuration"""
#
#    geom.wrap()
#    bohr2angs = 0.529177210670
#    coords = geom.get_positions()/bohr2angs
#    all_symbols = geom.get_chemical_symbols()
#    all_natoms = int(len(all_symbols))
#    
#    pseudocharge_dict = {}
#    for i in range(len(species)):
#        pseudocharge_dict[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering
#
#    # Compute unnormalized electron-density integral
#    iaux = 0
#    nele = 0.0
#    charge = 0.0
#    for iat in range(natoms):
#        spe = atomic_symbols[iat]
#        nele += pseudocharge_dict[spe]
#        for l in range(lmax[spe]+1):
#            for n in range(nmax[(spe,l)]):
#                if l==0:
#                    charge += charge_integrals[(spe,l,n)] * coefs[iaux]
#                iaux += 2*l+1
#
#    # Initialize dipole 
#    cart = ["y","z","x"]
#    dipole = {}
#    for icart in range(3):
#        dipole[cart[icart]] = 0.0
#    
#    # Perform dipole calculation
#    iaux = 0
#    for iat in range(all_natoms):
#        spe = all_symbols[iat]
#        if spe in species:
#            if average:
#                # Add contribution of nuclear pseudocharge to the dipole
#                dipole["x"] += pseudocharge_dict[spe] * coords[iat,0] 
#                dipole["y"] += pseudocharge_dict[spe] * coords[iat,1] 
#                dipole["z"] += pseudocharge_dict[spe] * coords[iat,2] 
#            for l in range(lmax[spe]+1):
#                for n in range(nmax[(spe,l)]):
#                    for im in range(2*l+1):
#                        if l==0:
#                            if average:
#                                # rescale isotropic coefficients to conserve the electronic charge
#                                coefs[iaux] *= nele/charge
#                            else:
#                                # remove residual charge from the most diffuse isotropic function
#                                if n==nmax[(spe,l)]-1:
#                                    coefs[iaux] -= charge/(charge_integrals[(spe,l,n)]*natoms)
#                            # Compute l=0 electronic contribution to the dipole 
#                            # NB: this is ill-defined in a truly periodic system and/or for systems with a net charge
#                            dipole["x"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,0]
#                            dipole["y"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,1]
#                            dipole["z"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
#                        if l==1:
#                            # Compute l=1 electronic contribution to the dipole 
#                            # NB: this follows the correspondence between (-1,0,1) real spherical harmonics and (y,z,x) Cartesian coordinates 
#                            dipole[cart[im]] -= coefs[iaux] * dipole_integrals[(spe,l,n)]
#                        iaux += 1
#
#    return [charge,dipole]

def read_local_pseudo(species, bdir):
    # Charge and radius of the local pseudopotential
    pseudocharge, rloc = {}, {}
    for spe in species:
        pp = np.loadtxt(osp.join(bdir, f"{spe}-local_pseudo.dat"))
        pseudocharge[spe] = pp[0]
        rloc[spe] = pp[1]
    return pseudocharge, rloc

def compute_charge_and_dipole(pseudocharge,natoms,atoms_range_set,atomic_symbols,coords,lmax,nmax,species,charge_integrals,dipole_integrals,coefs,average,parallel,comm):
    """Compute total charge and dipole moment for the given configuration"""

    pseudocharge_dict = {}
    for i in range(len(species)):
        pseudocharge_dict[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering

    # Compute unnormalized electron-density integral
    iaux = 0
    nele = 0.0
    charge = 0.0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        nele += pseudocharge_dict[spe]
        if iat in atoms_range_set:
            i = 0
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                        charge += charge_integrals[(spe,l,n)] * coefs[iaux+i]
                    i += 2*l+1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                iaux += 2*l+1

    if parallel:
        comm.Barrier()
        charge = comm.allreduce(charge)

    # Initialize dipole
    cart = ["y","z","x"]
    dipole = {}
    for icart in range(3):
        dipole[cart[icart]] = 0.0

    # Perform dipole calculation
    iaux = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if iat in atoms_range_set:
            i = 0
            if average:
                # Add contribution of nuclear pseudocharge to the dipole
                dipole["x"] += pseudocharge_dict[spe] * coords[iat,0]
                dipole["y"] += pseudocharge_dict[spe] * coords[iat,1]
                dipole["z"] += pseudocharge_dict[spe] * coords[iat,2]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0:
                            if average:
                                # rescale isotropic coefficients to conserve the electronic charge
                                coefs[iaux+i] *= nele/charge
                            else:
                                # remove residual charge from the most diffuse isotropic function
                                if n==nmax[(spe,l)]-1:
                                    coefs[iaux+i] -= charge/(charge_integrals[(spe,l,n)]*natoms)
                            # Compute l=0 electronic contribution to the dipole
                            # NB: this is ill-defined in a truly periodic system and/or for systems with a net charge
                            dipole["x"] -= coefs[iaux+i] * charge_integrals[(spe,l,n)] * coords[iat,0]
                            dipole["y"] -= coefs[iaux+i] * charge_integrals[(spe,l,n)] * coords[iat,1]
                            dipole["z"] -= coefs[iaux+i] * charge_integrals[(spe,l,n)] * coords[iat,2]
                        if l==1:
                            # Compute l=1 electronic contribution to the dipole
                            # NB: this follows the correspondence between (-1,0,1) real spherical harmonics and (y,z,x) Cartesian coordinates
                            dipole[cart[im]] -= coefs[iaux+i] * dipole_integrals[(spe,l,n)]
                        i += 1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                iaux += 2*l+1

    if parallel:
        comm.Barrier()
        dipole["x"] = comm.allreduce(dipole["x"])
        dipole["y"] = comm.allreduce(dipole["y"])
        dipole["z"] = comm.allreduce(dipole["z"])

    return [charge,dipole]

def scale_grad_coefs(pseudocharge,natoms,atoms_range_set,atomic_symbols,lmax,nmax,species,charge_integrals,coefs,grad_coefs,average,charge,parallel,comm):
    """Compute total charge and dipole moment for the given configuration"""

    pseudocharge_dict = {}
    for i in range(len(species)):
        pseudocharge_dict[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering

    # Compute unnormalized electron-density integral
    iaux = 0
    nele = 0.0
    grad_charge = np.zeros((all_natoms,3))
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        nele += pseudocharge_dict[spe]
        if iat in atoms_range_set:
            i=0
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                        grad_charge[:,:] += charge_integrals[(spe,l,n)] * grad_coefs[:,:,iaux+i]
                    i += 2*l+1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                iaux += 2*l+1

    #time_red_gc = time.time()
    if parallel:
        comm.Barrier()
        grad_charge = comm.allreduce(grad_charge)

    #print(f"Reduce time gc = {(time.time() - time_red_gc):.2f} s", flush=True)

    # Perform dipole calculation
    iaux = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if iat in atoms_range_set:
            i = 0
            if spe in species:
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        for im in range(2*l+1):
                            if l==0:
                                if average:
                                    # rescale isotropic coefficients to conserve the electronic charge
                                    grad_coefs[:,:,iaux + i] = (grad_coefs[:,:,iaux + i]*nele/charge)-coefs[iaux + i]*grad_charge[:,:]/charge
                                else:
                                    # remove residual charge from the most diffuse isotropic function
                                    if n==nmax[(spe,l)]-1:
                                        grad_coefs[:,:,iaux + i] -= grad_charge[:,:]/(charge_integrals[(spe,l,n)]*natoms)
                            i += 1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                iaux += 2*l+1

    return

def compute_polarizability(natoms,atomic_symbols,coords,lmax,nmax,species,charge_integrals,dipole_integrals,coefs):
    """Compute polarizability tensor for the given configuration"""

    # Compute unnormalized response integral
    charge = {} 
    for cart in ["x","y","z"]:
        ccoefs = coefs[cart]
        charge[cart] = 0.0
        iaux = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    if l==0:
                        charge[cart] += charge_integrals[(spe,l,n)] * ccoefs[iaux]
                    iaux += 2*l+1

    # Initialize polarizabilities 
    cart = ["y","z","x"]
    alpha = {}
    for cartrow in ["x","y","z"]:
        for icart in range(3):
            alpha[(cartrow,cart[icart])] = 0.0

    # Perform polarizability calculation
    for cartrow in ["x","y","z"]:
        ccoefs = coefs[cartrow]
        iaux = 0
        for iat in range(natoms):
            spe = atomic_symbols[iat]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0:
                            # remove residual charge from the most diffuse isotropic function
                            if n==nmax[(spe,l)]-1:
                                ccoefs[iaux] -= charge[cartrow]/(charge_integrals[(spe,l,n)]*natoms)
                            # Compute l=0 electronic contribution to the linear moment of the density-response 
                            # NB: this is ill-defined in a truly periodic system 
                            alpha[(cartrow,"x")] -= ccoefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,0]
                            alpha[(cartrow,"y")] -= ccoefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,1]
                            alpha[(cartrow,"z")] -= ccoefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                        if l==1:
                            # Compute l=1 electronic contribution to the linear moment of the density-response  
                            # NB: this follows the correspondence between (-1,0,1) real spherical harmonics and (y,z,x) Cartesian coordinates 
                            alpha[(cartrow,cart[im])] -= ccoefs[iaux] * dipole_integrals[(spe,l,n)]
                        iaux += 1

    return alpha

def get_basis_set_info_numba(lmax, nmax, species, dfbasis, bdir):
    # get basis set info 
    contra = Dict.empty(key_type=types.unicode_type,value_type=types.float64[:,:])
    alphas = Dict.empty(key_type=types.unicode_type,value_type=types.float64[:])
    npgf = Dict.empty(key_type=types.unicode_type,value_type=types.int64)
    contranorm = Dict.empty(key_type=types.unicode_type,value_type=types.float64[:,:])
    for spe in species:
        for l in range(lmax[spe]+1):
            key = f"{spe}_{l}"
            alphas[key] = np.atleast_1d(np.loadtxt(osp.join(bdir,f"{spe}-{dfbasis}-alphas-L{l}.dat")))
            contra[key] = np.atleast_2d(np.loadtxt(osp.join(bdir,f"{spe}-{dfbasis}-contra-L{l}.dat")))
            npgf[key] = alphas[key].shape[0]

    nbasis = Dict.empty(key_type=types.unicode_type,value_type=types.int64)
    nmax_numba = Dict.empty(key_type=types.unicode_type,value_type=types.int64)
    lmax_numba = Dict.empty(key_type=types.unicode_type,value_type=types.int64)

    for spe in species:
        nbasis[spe] = 0
        lmax_numba[spe] = lmax[spe]
        for l in range(lmax[spe]+1):
            key = f"{spe}_{l}"
            contranorm[key] = np.zeros((nmax[(spe,l)],npgf[key]))
            nmax_numba[key] = nmax[(spe,l)]
            for n in range(nmax_numba[key]):
                nbasis[spe] += 2*l+1
                inner = 0.0
                for ipgf1 in range(npgf[key]):
                    for ipgf2 in range(npgf[key]):
                        inner += contra[key][n,ipgf1] * contra[key][n,ipgf2] * 0.5 * math.gamma(l+1.5) / ( (alphas[key][ipgf1] + alphas[key][ipgf2])**(l+1.5) )
                sqrtinner = np.sqrt(inner)
                for ipgf in range(npgf[key]):
                    contranorm[key][n,ipgf] = contra[key][n,ipgf] / sqrtinner

    return lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm

def get_reciprocal_grid(nx, ny, nz, dx, dy, dz):

    # Get the G-vectors
    Gx = 2 * np.pi * np.fft.fftfreq(nx, dx)
    Gy = 2 * np.pi * np.fft.fftfreq(ny, dy)
    Gz = 2 * np.pi * np.fft.fftfreq(nz, dz)

    # Create full G-grid
    Gx_grid, Gy_grid, Gz_grid = np.meshgrid(Gx, Gy, Gz, indexing='ij')

    # Define a single array of G-vectors
    Gvec = np.stack((
        Gx_grid.ravel(),
        Gy_grid.ravel(),
        Gz_grid.ravel()
    ), axis=-1)

    return Gvec

@njit(parallel = True, fastmath = True)
def gto_rec(lmax,nmax,nbasis,species, npgf, contranorm, alphas, Gvec, nG_loc):

   partial_wave_coefs = Dict.empty(key_type=types.unicode_type,value_type=types.complex128[:,:]) # Dict with key as strings and values of type float array
   for spe in species:
      partial_wave_coefs[spe] = np.zeros((nG_loc, nbasis[spe]), dtype=np.complex128)

   for iG in prange(nG_loc):

      kx = Gvec[iG,0]
      ky = Gvec[iG,1]
      kz = Gvec[iG,2]

      # Norm squared |G|^2 and norm |G| of the k-mode vector
      knorm2 = kx*kx + ky*ky + kz*kz
      knorm = np.sqrt(knorm2)

      # Direction of G in spherical angles (costheta, phi)
      for spe in species:       
         if knorm == 0.0:
            costheta = 0.0
         else:
            costheta = kz/knorm
         phi = np.arctan2(ky,kx)
         
         ibasis = 0
         # Precompute partial wave coefficients <nlm|k> consisting in
         # spherical harmonics and radial integrals evaluated at the given k
         for lam in range(lmax[spe]+1):

            key = f"{spe}_{lam}"

            # Fourier transform prefactors
            lamfactor = np.sqrt(np.pi/2.0) * knorm**lam
            phase_lam = (-1.0j)**lam

            # Orthonormalized real spherical harmonics Y_{lam,m}(G/|G|) with Condon-Shortley phase convention
            harmonics = np.zeros((2*lmax[spe]+1))
            for mu in range(2*lam+1):
               harmonics[mu] = spherical_harmonic(lam, mu-lam, costheta, phi)
               
            # Primitive radial integrals
            pradintk = np.zeros((npgf[key]))
            for ipgf in range(npgf[key]):
               sigma2 = 1.0 / (2.0 * alphas[key][ipgf]) # Squared Gaussian width in reciprocal space
               sigma = np.sqrt(sigma2) # Gaussian width in reciprocal space
               pradintk[ipgf] = lamfactor * sigma2**lam * sigma**3.0 * np.exp(-0.5*knorm2*sigma2)

            # Precompute partial wave coefficients <nlm|k> consisting in
            # spherical harmonics and radial integrals evaluated at the given k
            radintk = np.zeros((max(nmax.values())))
            for irad in range(nmax[key]):
               radintk[irad] = 0.0
               for ipgf in range(npgf[key]):
                  radintk[irad] += contranorm[key][irad,ipgf]*pradintk[ipgf]
               for mu in range(2*lam+1):
                  partial_wave_coefs[spe][iG,ibasis] = radintk[irad] * harmonics[mu] * phase_lam
                  ibasis = ibasis + 1

   return partial_wave_coefs

@njit(parallel = True, fastmath = True)
def gto_rec_ewald(lmax,lcut,nmax,nbasis,species, npgf, contranorm, alphas, Gvec, nG_loc, sigma_ewald):

   partial_wave_coefs = Dict.empty(key_type=types.unicode_type,value_type=types.complex128[:,:]) # Dict with key as strings and values of type float array
   for spe in species:
      partial_wave_coefs[spe] = np.zeros((nG_loc, nbasis[spe]), dtype=np.complex128)

   for iG in prange(nG_loc):

      kx = Gvec[iG,0]
      ky = Gvec[iG,1]
      kz = Gvec[iG,2]

      # Norm squared |G|^2 and norm |G| of the k-mode vector
      knorm2 = kx*kx + ky*ky + kz*kz
      knorm = np.sqrt(knorm2)

      # Direction of G in spherical angles (costheta, phi)
      for spe in species:
         if knorm == 0.0:
            costheta = 0.0
         else:
            costheta = kz/knorm
         phi = np.arctan2(ky,kx)

         ibasis = 0
         # Precompute partial wave coefficients <nlm|k> consisting in
         # spherical harmonics and radial integrals evaluated at the given k
         for lam in range(min(lmax[spe]+1,lcut+1)):

            key = f"{spe}_{lam}"

            # Fourier transform prefactors
            lamfactor = np.sqrt(np.pi/2.0) * knorm**lam
            phase_lam = (-1.0j)**lam

            # Orthonormalized real spherical harmonics Y_{lam,m}(G/|G|) with Condon-Shortley phase convention
            harmonics = np.zeros((2*lmax[spe]+1))
            for mu in range(2*lam+1):
               harmonics[mu] = spherical_harmonic(lam, mu-lam, costheta, phi)

            # Primitive radial integrals
            pradintk = np.zeros((npgf[key]))
            for ipgf in range(npgf[key]):
               sigma2 = 1.0 / (2.0 * alphas[key][ipgf]) # Squared Gaussian width in reciprocal space
               sigma = np.sqrt(sigma2) # Gaussian width in reciprocal space
               pradintk[ipgf] = lamfactor * sigma2**lam * sigma**3.0 * np.exp(-0.5*knorm2*sigma_ewald**2)

            # Precompute partial wave coefficients <nlm|k> consisting in
            # spherical harmonics and radial integrals evaluated at the given k
            radintk = np.zeros((max(nmax.values())))
            for irad in range(nmax[key]):
               radintk[irad] = 0.0
               for ipgf in range(npgf[key]):
                  radintk[irad] += contranorm[key][irad,ipgf]*pradintk[ipgf]
               for mu in range(2*lam+1):
                  partial_wave_coefs[spe][iG,ibasis] = radintk[irad] * harmonics[mu] * phase_lam
                  ibasis = ibasis + 1

   return partial_wave_coefs

@njit(parallel = True, fastmath = True)
def gto_rec_prim(lmax, species, npgf, alphas, Gvec, gcuts):
    # Fourier transform of primitive atom-centered basis functions
    
    # Determine maximum relevant G-vector index
    gmax = 0
    for key in gcuts:
        gmax_key = gcuts[key].max()
        if gmax_key > gmax:
            gmax = gmax_key

    # Initialize partial wave coefficients
    partial_wave_coefs = Dict.empty(key_type=types.unicode_type, value_type=types.complex128[:, :, :]) # Dict with key as strings and values of type float array
    for spe in species:
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            # npgf[key]: number of individual Gaussians, each with its own alpha
            nmu = 2*lam + 1  #number of m values (-lam,...,+lam) for this lam
            partial_wave_coefs[key] = np.empty((gmax, npgf[key], nmu), dtype=np.complex128)

    # Loop over G-vectors and compute partial wave coefficients
    for iG in prange(gmax):

        kx = Gvec[iG, 0]
        ky = Gvec[iG, 1]
        kz = Gvec[iG, 2]

        # Norm squared |G|^2 and norm |G| of the k-mode vector
        knorm2 = kx*kx + ky*ky + kz*kz
        knorm = np.sqrt(knorm2)
        
        # Direction of G in spherical angles (costheta, phi)
        if knorm == 0.0:
            costheta = 0.0
            phi = 0.0
        else:
            costheta = kz/knorm
            phi = np.arctan2(ky, kx)

        for spe in species:
            # Precompute partial wave coefficients <nlm|k> consisting in
            # spherical harmonics and radial integrals evaluated at the given k
            for lam in range(lmax[spe]+1):

                key = f"{spe}_{lam}"

                # Fourier transform prefactors
                lamfactor = np.sqrt(np.pi/2.0) * knorm**lam
                phase_lam = (-1.0j)**lam

                # Orthonormalized real spherical harmonics Y_{lam,m}(G/|G|) with Condon-Shortley phase convention
                harmonics = np.zeros((2*lmax[spe]+1))
                for mu in range(2*lam+1):
                    harmonics[mu] = spherical_harmonic(lam, mu-lam, costheta, phi)

                for ipgf in range(npgf[key]):

                    if iG >= gcuts[key][ipgf]:
                        break # Do not calculate partial wave coefficients for G-vectors beyond the cutoff for this primitive

                    sigma2 = 1.0 / (2.0 * alphas[key][ipgf]) # Squared Gaussian width in reciprocal space
                    sigma = np.sqrt(sigma2) # Gaussian width in reciprocal space
                    
                    # Radial integral
                    radial = lamfactor * sigma2**lam * sigma**3.0 * np.exp(-0.5*knorm2*sigma2)
                    for mu in range(2*lam+1):
                        partial_wave_coefs[key][iG, ipgf, mu] = radial * harmonics[mu] * phase_lam

    return partial_wave_coefs

def gto_rec_g0(natoms, atomic_symbols, lmax, nmax, npgf, alphas, contranorm, ncoefs):
    # G=0 Fourier component (monopole) of each contracted basis function.
    # Needed for the Ewald G=0 correction.
    pwc_g0 = np.zeros(ncoefs)
    pwc_spread = np.zeros(ncoefs)

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            for irad in range(nmax[key]):
                for mu in range(2*lam+1):
                    if lam == 0:  # only l=0 has a nonzero monopole
                        for ipgf in range(npgf[key]):
                            sigma = 1.0 / np.sqrt(2.0 * alphas[key][ipgf])
                            pradintk = np.sqrt(np.pi/2) * sigma**3
                            harmonics = spherical_harmonic(0, 0, 0, 0) # 1/(2*sqrt(pi))
                            pwc_g0[icoefs] += contranorm[key][irad, ipgf] * pradintk * harmonics
                            pwc_spread[icoefs] += contranorm[key][irad, ipgf] * pradintk * harmonics * sigma**2
                    icoefs += 1
 
    return pwc_g0, pwc_spread

@njit
def spherical_harmonic(l,m,costheta,phi):
   # Compute orthonormalized real spherical harmonics

   if (m==0):
      normfactor = np.sqrt((2*l + 1) / (4.0*np.pi) )
      spherical_harmonic = normfactor * plgndr(l,0,costheta)

   elif (m<0):
      normfactor = np.sqrt( ((2*l + 1) / (4.0*np.pi) ) * factorial(l-abs(m))/factorial(l+abs(m)))
      spherical_harmonic = normfactor * plgndr(l,abs(m),costheta) * np.sqrt(2.0) * np.sin(abs(m)*phi)

   elif (m>0):
      normfactor = np.sqrt( ((2*l + 1) / (4.0*np.pi) ) * factorial(l-m)/factorial(l+m))
      spherical_harmonic = normfactor * plgndr(l,m,costheta) * np.sqrt(2.0) * np.cos(m*phi)

   # Condon-Shortley phase convention
   spherical_harmonic = spherical_harmonic * (-1.0)**m
   return spherical_harmonic

@njit
def plgndr(l,m,x):
   # Compute associate Legendre polynomials
   # Subroutine from Numerical Recipes in Fortran
   if (m<0 or m>l or abs(x)>1):
      print('ERROR: bad arguments in plgndr!')
      return 0
   pmm = 1.0
   if (m>0):
      somx2 = np.sqrt((1.0-x)*(1.0+x))
      fact = 1.0
      for i in range(m):
         pmm = -pmm*fact*somx2
         fact = fact + 2.0
   if (l==m):
      value=pmm
   else:
      pmmp1 = x*(2*m+1)*pmm
      if (l==m+1):
         value = pmmp1
      else:
         for ll in range(m+2,l+1):
            pll=(x*(2*ll-1)*pmmp1-(ll+m-1)*pmm)/(ll-m)
            pmm = pmmp1
            pmmp1 = pll
         value = pll
   
   return value

@njit
def factorial(n):
   if n < 0:
      print("Factorial is 0")
      return 0
   else:
      f = 1
      for i in range(1, n+1):
         f *= i
   return f

#@njit(parallel = False, fastmath = True)
def build_matrices(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, rho_KS_rec, nG_half, df_metric, rank):
    # Build the overlap matrix S fully in reciprocal space (DEPRECATED)
    # In order to compare the S matrix generated by this function with that of more recent implementations, multiply it by (4.0 * np.pi) / volume
    S = np.zeros((ncoefs, ncoefs), dtype=np.float64)
    w = np.zeros((ncoefs), dtype=np.float64)

    knorm_vec = np.sqrt(np.sum(Gvec_half*Gvec_half,axis=1)).astype(np.float64)

    cos_k_coords = np.cos(np.dot(Gvec_half,coords.T))
    sin_k_coords = np.sin(np.dot(Gvec_half,coords.T))
    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        obj1 = np.zeros((nG_half, nbasis[spe]), dtype = np.complex128)
        if df_metric == "identity":
            obj1 = (cos_k_coords - 1j * sin_k_coords)[:, iat, np.newaxis] * partial_wave_coefs[spe]
        if df_metric == "coulomb":
            obj1 = ((cos_k_coords - 1j * sin_k_coords)/ knorm_vec[:, np.newaxis])[:, iat, np.newaxis] * partial_wave_coefs[spe]
        icoefs2 = 0
        for iat2 in range(iat+1):
            spe2 = atomic_symbols[iat2]
            obj2 = np.zeros((nG_half, nbasis[spe2]), dtype = np.complex128)
            if df_metric == "identity":
                obj2 = (cos_k_coords - 1j * sin_k_coords)[:, iat2, np.newaxis] * partial_wave_coefs[spe2]
            if df_metric == "coulomb":
                obj2 = ((cos_k_coords - 1j * sin_k_coords)/ knorm_vec[:, np.newaxis] )[:, iat2, np.newaxis] * partial_wave_coefs[spe2]

            if df_metric == "identity":
                S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]] = 2*(np.dot(obj1[1:,:].real.T, obj2[1:,:].real) + np.dot(obj1[1:,:].imag.T, obj2[1:,:].imag))
                S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]] += np.outer(obj1[0,:].real, obj2[0,:].real) + np.outer(obj1[0,:].imag, obj2[0,:].imag)
                if icoefs != icoefs2:
                    S[icoefs2:icoefs2+nbasis[spe2], icoefs:icoefs+nbasis[spe]] = S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]].T
            if df_metric == "coulomb":
                S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]] = 2*(np.dot(obj1.real.T, obj2.real) + np.dot(obj1.imag.T, obj2.imag))
                S[icoefs2:icoefs2+nbasis[spe2], icoefs:icoefs+nbasis[spe]] = S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]].T
            icoefs2 += nbasis[spe2]
        if df_metric == "identity":
            w[icoefs:icoefs+nbasis[spe]] = 2*(np.dot(obj1[1:,:].real.T, rho_KS_rec[1:].real) + np.dot(obj1[1:,:].imag.T, rho_KS_rec[1:].imag))
            w[icoefs:icoefs+nbasis[spe]] += obj1[0,:].real.T * rho_KS_rec[0].real + obj1[0,:].imag.T * rho_KS_rec[0].imag
        if df_metric == "coulomb":
            w[icoefs:icoefs+nbasis[spe]] = 2*(np.dot(obj1.real.T, rho_KS_rec.real/knorm_vec) + np.dot(obj1.imag.T, rho_KS_rec.imag/knorm_vec))
        icoefs += nbasis[spe]

    S = np.real(S)*4*np.pi
    w = np.real(w)

    return S, w

#@njit(parallel = False, fastmath = True)
def build_matrices_prim(Gvec_half, natoms, coords, npgf, lmax, atomic_symbols, partial_wave_coefs, df_metric, ncut, rank):
    # Build the primitive-basis overlap matrix Sp in reciprocal space (DEPRECATED)
    # In order to compare the S matrix generated by this function with that of more recent implementations, multiply it by (4.0 * np.pi) / volume
    offset = {}
    ipgf = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            for mu in range(2*lam+1):
                offset[(iat, lam, mu)] = ipgf
                ipgf += npgf[key]
    ncoefs_prim = ipgf

    Sp = np.zeros((ncoefs_prim, ncoefs_prim), dtype=np.float64)

    knorm_vec = np.sqrt(np.sum(Gvec_half*Gvec_half,axis=1)).astype(np.float64) # |G|
    phase = np.exp(-1j * np.dot(Gvec_half, coords.T)) # e^{-iG.r_iat}

    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for iat2 in range(iat+1):
            spe2 = atomic_symbols[iat2]
            same_atom = (iat2 == iat)

            phase_pair = phase[:, iat] * np.conj(phase[:, iat2])
            if df_metric == "coulomb":
                phase_pair = phase_pair / (knorm_vec**2)

            phase_pair_w = 2.0 * phase_pair
            if df_metric == "identity":
                phase_pair_w[0] = phase_pair[0]

            phase_pair_w_real = phase_pair_w.real
            phase_pair_w_imag = phase_pair_w.imag

            for lam in range(lmax[spe]+1):
                key = f"{spe}_{lam}"
                for mu in range(2*lam+1):
                    row_base = offset[(iat, lam, mu)]

                    for lam2 in range(lmax[spe2]+1):
                        if same_atom and lam2 < lam:
                            continue  # this (lam2,lam) block already done from the other side
                        key2 = f"{spe2}_{lam2}"
                        for mu2 in range(2*lam2+1):
                            if same_atom and lam2 == lam and mu2 < mu:
                                continue
                            col_base = offset[(iat2, lam2, mu2)]

                            for ipgf1 in range(npgf[key]):
                                n1 = ncut[key][ipgf1]
                                pwc1 = partial_wave_coefs[key][:, ipgf1, mu]

                                for ipgf2_local in range(npgf[key2]):
                                    if same_atom and lam2 == lam and mu2 == mu and ipgf2_local < ipgf1:
                                        continue
                                    n2 = ncut[key2][ipgf2_local]
                                    ncut_pair = min(n1, n2)

                                    pwc2 = partial_wave_coefs[key2][:ncut_pair, ipgf2_local, mu2]
                                    c = pwc1[:ncut_pair] * np.conj(pwc2)

                                    val = (np.dot(phase_pair_w_real[:ncut_pair], c.real) - np.dot(phase_pair_w_imag[:ncut_pair], c.imag))

                                    row = row_base + ipgf1
                                    col = col_base + ipgf2_local
                                    Sp[row, col] = val
                                    if row != col:
                                        Sp[col, row] = val

    Sp = np.real(Sp) * 4 * np.pi
    return Sp

def get_w_prim(Gvec_half, natoms, coords, npgf, lmax, atomic_symbols, partial_wave_coefs, volume, rho_KS_rec, df_metric, gcuts, rank):
    # Build the primitive-basis density vector wp
    
    # Get the total size of the primitive basis
    ncoefs_prim = 0
    gmax = 0
    gmax_key = {}
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            ncoefs_prim += npgf[key] * (2*lam + 1)
            gmax_key[key] = gcuts[key].max()
            gmax = max(gmax, gmax_key[key])

    wp = np.zeros((ncoefs_prim), dtype=np.float64)

    Gvec_half = Gvec_half[:gmax]
    knorm2_vec = np.einsum('ij,ij->i', Gvec_half, Gvec_half) # |G|^2
    
    # Precompute the G-weighting
    if df_metric == "identity":
        # weight 2 for G>0, weight 1 for G=0
        rho_w = 2.0 * np.conj(rho_KS_rec[:gmax])
        rho_w[0] = np.conj(rho_KS_rec[0])
    if df_metric == "coulomb":
        # weight always 2 here since Gvec_half already excludes G=0
        rho_w = 2.0 * np.conj(rho_KS_rec[:gmax]) * (4.0 * np.pi) / knorm2_vec
    
    # Precompute real/imag parts of partial_wave_coefs ONCE per key
    pwc_real = {key: partial_wave_coefs[key][:gmax_key[key],:,:].real for key in partial_wave_coefs}
    pwc_imag = {key: partial_wave_coefs[key][:gmax_key[key],:,:].imag for key in partial_wave_coefs}
    
    ipgf = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]

        nmax_atom = 0
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            nmax_atom = max(nmax_atom, gmax_key[key])

        z = np.exp(-1j * np.dot(Gvec_half[:nmax_atom], coords[iat])) * rho_w[:nmax_atom]
        z_real = np.ascontiguousarray(z.real)
        z_imag = np.ascontiguousarray(z.imag)

        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"

            for mu in range(2*lam + 1):
                for ipgf1 in range(npgf[key]):
                    n1 = gcuts[key][ipgf1]
                    wp[ipgf+ipgf1] = (np.dot(pwc_real[key][:n1, ipgf1, mu], z_real[:n1]) - np.dot(pwc_imag[key][:n1, ipgf1, mu], z_imag[:n1]))

                ipgf += npgf[key]

    wp = wp * (4.0 * np.pi) / volume

    return wp

def setup_pyscf_species(species, lmax, nmax, alphas, contranorm):
    #Build per-species PySCF Mole objects

    mol_bra = {} # fixed (bra) side of overlap integrals
    mol_ket = {} # movable (ket) side of overlap integrals
    coord_slice = {} # ket (x,y,z) coordinate
    perm = {}

    for spe in species:
        basis = []
        idx = []
        off = 0
        for lam in range(lmax[spe] + 1):
            key = f"{spe}_{lam}"
            shell = [lam] # The first entry tells PySCF the angular momentum

            for ipgf in range(len(alphas[key])):
                # Each row expected by PySCF looks like:
                #   [exponent, coeff_for_function_0, coeff_for_function_1, ...]
                row = [float(alphas[key][ipgf])]
                for irad in range(nmax[key]):
                    row.append(float(contranorm[key][irad, ipgf]) / _pyscf_gto.gto_norm(lam, float(alphas[key][ipgf]))) # divide by PySCF's normalization factor (gto_norm) to avoid PySCF's automatic normalization of the primitive
                shell.append(row)

            basis.append(shell) # One entry per angular momentum shell

            # Index map to fix ordering mismatch between PySCF and SALTED (only matters for lam=1)
            mmap = [1, 2, 0] if lam == 1 else list(range(2 * lam + 1))
            for irad in range(nmax[key]):
                for mu in range(2 * lam + 1):
                    idx.append(off + irad * (2 * lam + 1) + mmap[mu])
            off += nmax[key] * (2 * lam + 1)

        # PySCF molecule object builder
        # "Ghost" atom sitting at the origin, carrying the basis assembled above.
        def _make():
            mol = _pyscf_gto.M(
                atom=[[f"ghost-{spe}", (0.0, 0.0, 0.0)]],
                basis={f"ghost-{spe}": basis},
                spin=0,
                cart=False,    # real spherical harmonics, not cartesian Gaussians
                unit="Bohr",   # match SALTED units
            )
            return mol

        mol_bra[spe] = _make()
        mol_ket[spe] = _make()
        ptr = mol_ket[spe]._atm[0, _pyscf_gto.PTR_COORD]
        coord_slice[spe] = slice(ptr, ptr + 3)
        perm[spe] = np.asarray(idx, dtype=int)

    return {"mol_bra": mol_bra, "mol_ket": mol_ket, "perm": perm, "coord_slice": coord_slice}

def setup_pyscf_core(species, pseudocharge, rloc):
    # Build per-species PySCF Mole objects for the Gaussian core charge distribution (a single s function)

    mol_core = {}
    coord_slice = {}
    scale = {}

    for spe in species:
        sigma = rloc[spe]
        alpha = 1.0 / (2.0 * sigma**2)

        # "Ghost" atom sitting at the origin, carrying the basis assembled above
        mol = _pyscf_gto.M(
            atom=[[f"ghost-{spe}", (0.0, 0.0, 0.0)]],
            basis={f"ghost-{spe}": [[0, [float(alpha), 1.0]]]},
            spin=0,
            cart=False,    # real spherical harmonics, not cartesian Gaussians
            unit="Bohr",   # match SALTED units
        )

        # Rescale to have the correct pseudocharge
        mol_core[spe] = mol
        scale[spe] = (pseudocharge[spe] / ((2.0*np.pi)**1.5 * sigma**3)) / mol.eval_gto("GTOval_sph", np.zeros((1,3)))[0,0]
        ptr = mol._atm[0, _pyscf_gto.PTR_COORD]
        coord_slice[spe] = slice(ptr, ptr + 3)

    return {"mol_core": mol_core, "coord_slice": coord_slice, "scale": scale}

   # add next to the existing pyscf import

def setup_pyscf_ewald(species, lmax, nmax, alphas, contranorm, sigma_ewald):
    # Build per-species PySCF Mole objects carrying the Ewald-corrected radial functions
    #     R_nl(r) - R_nl^ewald(r),   R_nl^ewald(r) = d_nl * r^lam * exp(-alpha_ewald*r^2)

    alpha_ewald = 1.0 / (2.0 * sigma_ewald**2)

    mol_bra = {} # fixed (bra) side of overlap integrals
    mol_ket = {} # movable (ket) side of overlap integrals
    coord_slice = {} # ket (x,y,z) coordinate
    perm = {}

    for spe in species:
        basis = []
        idx = []
        off = 0
        for lam in range(lmax[spe] + 1):
            key = f"{spe}_{lam}"
            shell = [lam] # The first entry tells PySCF the angular momentum

            for ipgf in range(len(alphas[key])):
                # Each row expected by PySCF looks like:
                #   [exponent, coeff_for_function_0, coeff_for_function_1, ...]
                row = [float(alphas[key][ipgf])]
                for irad in range(nmax[key]):
                    row.append(float(contranorm[key][irad, ipgf]) / _pyscf_gto.gto_norm(lam, float(alphas[key][ipgf]))) # divide by PySCF's normalization factor (gto_norm) to avoid PySCF's automatic normalization of the primitive
                shell.append(row)

            # Extra primitive carrying -R_nl^ewald
            row = [float(alpha_ewald)]
            for irad in range(nmax[key]):
                dnl = np.sum(contranorm[key][irad, :] * (alpha_ewald / alphas[key])**(lam + 1.5))
                row.append(-float(dnl) / _pyscf_gto.gto_norm(lam, float(alpha_ewald)))
            shell.append(row)
            
            basis.append(shell) # One entry per angular momentum shell

            # Index map to fix ordering mismatch between PySCF and SALTED (only matters for lam=1)
            mmap = [1, 2, 0] if lam == 1 else list(range(2 * lam + 1))
            for irad in range(nmax[key]):
                for mu in range(2 * lam + 1):
                    idx.append(off + irad * (2 * lam + 1) + mmap[mu])
            off += nmax[key] * (2 * lam + 1)

        # PySCF molecule object builder
        # "Ghost" atom sitting at the origin, carrying the basis assembled above.
        def _make():
            normalize_gto = _pyscf_gto.mole.NORMALIZE_GTO
            _pyscf_gto.mole.NORMALIZE_GTO = False # disable normalization to build R_nl - R_nl^ewald
            mol = _pyscf_gto.M(
                atom=[[f"ghost-{spe}", (0.0, 0.0, 0.0)]],
                basis={f"ghost-{spe}": basis},
                spin=0,
                cart=False,    # real spherical harmonics, not cartesian Gaussians
                unit="Bohr",   # match SALTED units
            )
            _pyscf_gto.mole.NORMALIZE_GTO = normalize_gto
            return mol

        mol_bra[spe] = _make()
        mol_ket[spe] = _make()
        ptr = mol_ket[spe]._atm[0, _pyscf_gto.PTR_COORD]
        coord_slice[spe] = slice(ptr, ptr + 3)
        perm[spe] = np.asarray(idx, dtype=int)

    return {"mol_bra": mol_bra, "mol_ket": mol_ket, "perm": perm, "coord_slice": coord_slice}

def pair_cutoffs(species, lmax, alphas, contranorm, eps=1.0e-12):
    #Real-space cutoff (Bohr) per pair
    amin, cmax = {}, {}

    for spe in species:
        amin[spe] = min(float(np.min(alphas[f"{spe}_{lam}"])) for lam in range(lmax[spe] + 1))
        cmax[spe] = max(float(np.max(np.abs(contranorm[f"{spe}_{lam}"]))) for lam in range(lmax[spe] + 1))
    rcut = {}

    for spe1 in species:
        for spe2 in species:
            #amin_pair = amin[spe1] * amin[spe2] / (amin[spe1] + amin[spe2])
            A = cmax[spe1] * cmax[spe2] * (np.pi / (amin[spe1] + amin[spe2])) ** 1.5
            smax1 = np.sqrt((2.0 + lmax[spe1]) / (2.0 * amin[spe1]))
            smax2 = np.sqrt((2.0 + lmax[spe2]) / (2.0 * amin[spe2]))
            r = 4.0 * (smax1 + smax2)
            #r = 4.0 * np.sqrt(1.0 / (2.0 * amin_pair)) # Initial guess: r_cut = 4*sigma_pair
            #r = np.sqrt(max(np.log(max(A, 1.0) * max(r, 1.0) ** (lmax[spe1] + lmax[spe2]) / eps), 1.0) / amin_pair) # Optional refinement: fixed-point iteration
            rcut[(spe1, spe2)] = r
    return rcut

def lattice_images(cell, rcut_max, volume):

    cell = np.asarray(cell, dtype=float) # cell: (3,3) array, each row is one lattice vector (a1, a2, a3) in Bohr.

    h = np.empty(3)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3 # cyclic: 0,1,2 -> 1,2,0 -> 2,0,1
        face_area = np.linalg.norm(np.cross(cell[j], cell[k])) # |cell[j] x cell[k]| is the area of the parallelogram face spanned by lattice vectors j and k
        h[i] = volume / face_area # how far apart are the periodic copies of the cell

    nmax = np.ceil(rcut_max / h).astype(int) + 1 # Number of periodic images to include along each lattice vector direction

    grids = [np.arange(-n, n + 1) for n in nmax]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    ns = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], axis=1) 

    return ns @ cell # Return converted to a Cartesian translation

def overlap_identity(cell, coords, atomic_symbols, nbasis, ncoefs, volume, pyscf_data, rcut):
    #Identity-metric overlap matrix in real space

    natoms = len(atomic_symbols)

    S = np.zeros((ncoefs, ncoefs))

    # Unpack PySCF data
    mol_bra = pyscf_data["mol_bra"]
    mol_ket = pyscf_data["mol_ket"]
    perm = pyscf_data["perm"]
    cslice = pyscf_data["coord_slice"]

    rcut_max = max(rcut.values()) # Pairs cutoff, to decided how many periodic images to include
    periodic = cell is not None and volume > 1.0e-10
    if periodic:
        cell = np.asarray(cell, dtype=float)
        inv_cell = np.linalg.inv(cell) # Cartesian to fractional coordinates
        images = lattice_images(cell, rcut_max, volume) # Bring periodic images
    else:
        images = np.zeros((1, 3)) # No periodicity

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        pbra = perm[spe]

        icoefs2 = 0
        for iat2 in range(iat + 1):
            spe2 = atomic_symbols[iat2]
            pket = perm[spe2]

            rcut_ij = rcut[(spe, spe2)] # real-space cutoff to use for this specific pair

            delta = coords[iat2] - coords[iat]

            if periodic:
                frac = delta @ inv_cell # Convert to fractional coordinates
                delta = (frac - np.round(frac)) @ cell # Wrap and convert back to Cartesian

            dvecs = delta + images
            keep = np.einsum("ij,ij->i", dvecs, dvecs) <= rcut_ij * rcut_ij # Keep only the images whose distance falls within the pair cutoff

            block = np.zeros((nbasis[spe], nbasis[spe2]))
            for d in dvecs[keep]:
                mol_ket[spe2]._env[cslice[spe2]] = d # Move the "ket" ghost atom
                raw = _pyscf_gto.intor_cross("int1e_ovlp", mol_bra[spe], mol_ket[spe2]) # Ask PySCF for the raw overlap integrals
                block += raw[np.ix_(pbra, pket)] # Reorder PySCF's rows/columns into SALTED's (n,m) ordering

            S[icoefs:icoefs + nbasis[spe], icoefs2:icoefs2 + nbasis[spe2]] = block
            if iat2 != iat:
                S[icoefs2:icoefs2 + nbasis[spe2], icoefs:icoefs + nbasis[spe]] = block.T

            icoefs2 += nbasis[spe2]
        icoefs += nbasis[spe]

    return S

def overlap_coulomb_real(cell, coords, atomic_symbols, nbasis, ncoefs, volume, pyscf_data, pyscf_ewald, pwc_g0, pwc_spread, sigma_ewald, rcut):
    #Identity-metric overlap matrix in real space

    natoms = len(atomic_symbols)

    S = np.zeros((ncoefs, ncoefs))

    # Unpack PySCF data
    mol_bra = pyscf_data["mol_bra"]
    mol_ket = pyscf_ewald["mol_ket"]
    perm = pyscf_data["perm"]
    cslice = pyscf_data["coord_slice"]

    rcut_max = max(rcut.values()) # Pairs cutoff, to decided how many periodic images to include
    periodic = cell is not None and volume > 1.0e-10
    if periodic:
        cell = np.asarray(cell, dtype=float)
        inv_cell = np.linalg.inv(cell) # Cartesian to fractional coordinates
        images = lattice_images(cell, rcut_max, volume) # Bring periodic images
    else:
        images = np.zeros((1, 3)) # No periodicity

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        pbra = perm[spe]

        icoefs2 = 0
        for iat2 in range(iat + 1):
            spe2 = atomic_symbols[iat2]
            pket = perm[spe2]

            rcut_ij = rcut[(spe, spe2)] # real-space cutoff to use for this specific pair

            delta = coords[iat2] - coords[iat]

            if periodic:
                frac = delta @ inv_cell # Convert to fractional coordinates
                delta = (frac - np.round(frac)) @ cell # Wrap and convert back to Cartesian

            dvecs = delta + images
            keep = np.einsum("ij,ij->i", dvecs, dvecs) <= rcut_ij * rcut_ij # Keep only the images whose distance falls within the pair cutoff

            block = np.zeros((nbasis[spe], nbasis[spe2]))
            for d in dvecs[keep]:
                mol_ket[spe2]._env[cslice[spe2]] = d # Move the "ket" ghost atom
                raw = _pyscf_gto.intor_cross("int2c2e", mol_bra[spe], mol_ket[spe2]) # Ask PySCF for the raw overlap integrals
                block += raw[np.ix_(pbra, pket)] # Reorder PySCF's rows/columns into SALTED's (n,m) ordering

            # G=0 correction
            a_bra = pwc_g0[icoefs:icoefs + nbasis[spe]]
            a_ket = pwc_g0[icoefs2:icoefs2 + nbasis[spe2]]
            b_ket = pwc_spread[icoefs2:icoefs2 + nbasis[spe2]]
            block -= (4.0*np.pi)**3 / (2.0*volume) * np.outer(a_bra, sigma_ewald**2 * a_ket - b_ket)

            S[icoefs:icoefs + nbasis[spe], icoefs2:icoefs2 + nbasis[spe2]] = block
            if iat2 != iat:
                S[icoefs2:icoefs2 + nbasis[spe2], icoefs:icoefs + nbasis[spe]] = block.T

            icoefs2 += nbasis[spe2]
        icoefs += nbasis[spe]

    return S

def overlap_coulomb_rec(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, partial_wave_coefs_ewald, volume, gcut, rank):
    # Long-range Coulomb metric in reciprocal space: (Phi_i | R_j^ewald)
    S = np.zeros((ncoefs, ncoefs), dtype=np.float64)

    # G-vector truncation based on sigma_ewald
    Gvec_half = Gvec_half[:gcut]
    knorm_vec = np.sqrt(np.sum(Gvec_half*Gvec_half, axis=1)).astype(np.float64)  # |G|

    phase = np.exp(-1j * np.dot(Gvec_half, coords.T)) # e^{-iG.r_iat}
    phase_over_knorm = phase / knorm_vec[:, np.newaxis]

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        obj1 = phase_over_knorm[:, iat, np.newaxis] * partial_wave_coefs[spe][:gcut]
        obj1_real = np.ascontiguousarray(obj1.real)
        obj1_imag = np.ascontiguousarray(obj1.imag)

        icoefs2 = 0
        for iat2 in range(iat+1):
            spe2 = atomic_symbols[iat2]
            obj2 = phase_over_knorm[:, iat2, np.newaxis] * partial_wave_coefs_ewald[spe2][:gcut]  # Ewald ket

            S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]] = 2*(np.dot(obj1_real.T, obj2.real) + np.dot(obj1_imag.T, obj2.imag))
            if iat2 != iat:
                S[icoefs2:icoefs2+nbasis[spe2], icoefs:icoefs+nbasis[spe]] = S[icoefs:icoefs+nbasis[spe], icoefs2:icoefs2+nbasis[spe2]].T
            icoefs2 += nbasis[spe2]
        icoefs += nbasis[spe]

    S = S * (4.0 * np.pi)**3 / volume

    return S

def build_contraction_matrix(natoms, atomic_symbols, lmax, nmax, npgf, contranorm):
    ncoefs_prim = 0
    ncoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            ncoefs_prim += npgf[key] * (2*lam + 1)
            ncoefs      += nmax[key] * (2*lam + 1)

    C = np.zeros((ncoefs_prim, ncoefs), dtype=np.float64)

    ipgf = 0
    icoef = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            for irad in range(nmax[key]):
                for mu in range(2*lam + 1):
                    row_start = ipgf + mu*npgf[key]
                    col = icoef + irad*(2*lam+1) + mu
                    C[row_start:row_start+npgf[key], col] = contranorm[key][irad, :]

            ipgf  += npgf[key]*(2*lam+1)
            icoef += nmax[key]*(2*lam+1)

    return C

def gmax_for_prim(alpha):
    sigma = np.sqrt(1.0 / (2.0 * alpha))
    return 2 * np.pi / sigma

def build_gcutoff(alphas, npgf, species, lmax, knorm_vec, nG_half):
    gcuts = Dict.empty(key_type=types.unicode_type, value_type=types.int64[:])
    for spe in species:
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            gmax = gmax_for_prim(alphas[key])
            gcuts[key] = np.searchsorted(knorm_vec, gmax).astype(np.int64) #Find index where Gmax would be inserted to to maintain order.
            #gcuts[key] = np.full(npgf[key], nG_half, dtype=np.int64) # For debugging purposes, set gcuts to the full size of G.
    return gcuts

def elec_energy_forces(lmax,nmax,saltedpath,dfbasis,species,structure,coefs):

    bdir = osp.join(saltedpath,"basis")
    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, dfbasis, bdir)

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
    natoms = len(atomic_symbols)
    coords  = structure.positions/b2a
    volume = structure.get_volume()/(b2a**3)

    nx = int(np.floor(structure.cell[0,0]/(0.111*b2a))+1)
    ny = int(np.floor(structure.cell[1,1]/(0.111*b2a))+1)
    nz = int(np.floor(structure.cell[2,2]/(0.111*b2a))+1)

    dx, dy, dz = structure.cell[0,0]/(b2a*nx), structure.cell[1,1]/(b2a*ny), structure.cell[2,2]/(b2a*nz)

    Gvec = get_reciprocal_grid(nx,ny,nz,dx,dy,dz)

    mask = (
    (Gvec[:, 2] > 0) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0))
    )

    Gvec_half = Gvec[mask][1:]  # Exclude G=0

    nG_half=len(Gvec_half)


    #time_pwc = time.time()
    partial_wave_coefs = gto_rec(lmax_numba,nmax_numba,nbasis,species,npgf, contranorm, alphas,Gvec_half, nG_half)

    #print(time.time()-time_pwc)

    cos_k_coords = np.cos(np.dot(Gvec_half,coords.T))
    sin_k_coords = np.sin(np.dot(Gvec_half,coords.T))

    knorm2_vec = np.sum(Gvec_half*Gvec_half,axis=1)

    gauss = {}
    for spe in species:
       gauss[spe] = np.exp(-0.5*knorm2_vec*(rloc_dict[spe]**2))

    volfactor = 32.0*np.pi*np.pi/(volume)

    offset = 0
    rho_rec = np.zeros((nG_half, natoms), dtype=np.complex128)
    rho_n_rec = np.zeros((nG_half, natoms), dtype=np.complex128)
    forces = np.zeros((natoms,3), dtype=np.complex128)

    #time_coefs_dot = time.time()

    for iat in range(natoms):
       spe = atomic_symbols[iat]
       rho_rec[:,iat] = -np.dot(partial_wave_coefs[spe],coefs[offset:offset + nbasis[spe]]) * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat])
       rho_n_rec[:,iat] = +pseudocharge_numba[spe] * gauss[spe] * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat])
       offset += nbasis[spe]

    #print(time.time()-time_coefs_dot)

    time_energy = time.time()

    U_tot = np.dot(np.sum((rho_n_rec/(4*np.pi)) + rho_rec, axis = 1)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)), 1/knorm2_vec)
    
    forces[:,0] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,0][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,0][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))
    forces[:,1] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,1][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,1][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))
    forces[:,2] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,2][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,2][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))

    #print(time.time()-time_energy)

    U_tot = np.real(U_tot * 2*np.pi * volfactor)
    forces = np.real(forces * 2*np.pi * volfactor)

    return U_tot, forces

def elec_energy_forces_ewald(lmax,lcut,nmax,saltedpath,dfbasis,species,pseudocharge,rloc_dict,structure,coefs):

    b2a = 0.529177249

    bdir = osp.join(saltedpath,"basis")

    sigma_ewald = 1.0/b2a

    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, dfbasis, bdir)

    pseudocharge_numba = Dict.empty(key_type=types.unicode_type,value_type=types.float64)
    for i in range(len(species)):
       pseudocharge_numba[species[i]] = pseudocharge[i] # Warning: species and pseudocharge must have the same ordering

    atomic_symbols = structure.get_chemical_symbols()
    natoms = len(atomic_symbols)
    coords  = structure.positions/b2a

    volume = structure.get_volume()/(b2a**3)

    nx = int(np.floor(structure.cell[0,0]/((np.pi*sigma_ewald/(2*np.sqrt(2)))*b2a))+1)
    ny = int(np.floor(structure.cell[1,1]/((np.pi*sigma_ewald/(2*np.sqrt(2)))*b2a))+1)
    nz = int(np.floor(structure.cell[2,2]/((np.pi*sigma_ewald/(2*np.sqrt(2)))*b2a))+1)

    dx, dy, dz = structure.cell[0,0]/(b2a*nx), structure.cell[1,1]/(b2a*ny), structure.cell[2,2]/(b2a*nz)

    Gvec = get_reciprocal_grid(nx,ny,nz,dx,dy,dz)

    mask = (
    (Gvec[:, 2] > 0) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0)) |
    ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0))
    )

    Gvec_half = Gvec[mask][1:]  # Exclude G=0

    nG_half=len(Gvec_half)


    #time_pwc = time.time()
    partial_wave_coefs = gto_rec_ewald(lmax_numba,lcut,nmax_numba,nbasis,species,npgf, contranorm, alphas,Gvec_half, nG_half, sigma_ewald)

    #print(time.time()-time_pwc)

    cos_k_coords = np.cos(np.dot(Gvec_half,coords.T))
    sin_k_coords = np.sin(np.dot(Gvec_half,coords.T))

    knorm2_vec = np.sum(Gvec_half*Gvec_half,axis=1)

    gauss = {}
    for spe in species:
       gauss[spe] = np.exp(-0.5*knorm2_vec*(sigma_ewald**2))

    volfactor = 32.0*np.pi*np.pi/(volume)

    offset = 0
    rho_rec = np.zeros((nG_half, natoms), dtype=np.complex128)
    rho_n_rec = np.zeros((nG_half, natoms), dtype=np.complex128)
    forces = np.zeros((natoms,3), dtype=np.complex128)

    #time_coefs_dot = time.time()

    for iat in range(natoms):
       spe = atomic_symbols[iat]
       rho_rec[:,iat] = -np.dot(partial_wave_coefs[spe],coefs[offset:offset + nbasis[spe]]) * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat])
       rho_n_rec[:,iat] = +pseudocharge_numba[spe] * gauss[spe] * (cos_k_coords[:, iat] - 1j * sin_k_coords[:, iat])
       offset += nbasis[spe]

    #print(time.time()-time_coefs_dot)

    time_energy = time.time()

    U_tot = np.dot(np.sum((rho_n_rec/(4*np.pi)) + rho_rec, axis = 1)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)), 1/knorm2_vec)
    
    forces[:,0] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,0][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,0][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))
    forces[:,1] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,1][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,1][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))
    forces[:,2] = np.dot(1/knorm2_vec, (1j *Gvec_half[:,2][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))+np.conj((1j *Gvec_half[:,2][:, np.newaxis]*((rho_n_rec/(4*np.pi)) + rho_rec)*np.conj(np.sum((rho_n_rec/(4*np.pi))+rho_rec,axis = 1)[:, np.newaxis]))))

    #print(time.time()-time_energy)

    U_tot = np.real(U_tot * 2*np.pi * volfactor)
    forces = np.real(forces * 2*np.pi * volfactor)

    return U_tot, forces

def get_rho_n(nside, dx, dy, dz, origin, natoms, coords, atomic_symbols, pseudocharge, rloc, nsigma=6.0):
    # Gaussian core charge on the cube grid

    rho_n = np.zeros((nside[0], nside[1], nside[2]), dtype=np.float64)
    dr = np.array([dx, dy, dz]) # grid spacing along x, y, z

    for iat in range(natoms):
        spe = atomic_symbols[iat]
        sigma = rloc[spe] # Gaussian widths
        rcut = nsigma * sigma # radius beyond which the Gaussian is negligible (default 6 sigma)

        gauss = [] # List of 1D Gaussians along each axis
        idx = [] # List of grid indices along each axis

        for i in range(3): # Loop over x, y, z
            imin = int(np.floor((coords[iat,i] - origin[i] - rcut)/dr[i]))
            imax = int(np.ceil((coords[iat,i] - origin[i] + rcut)/dr[i]))
            ivec = np.arange(imin, imax+1)
            r = origin[i] + ivec*dr[i] - coords[iat,i] # Physical position of the grid points relative to the atom position

            gauss.append(np.exp(-0.5*(r/sigma)**2)) # 1D Gaussian
            idx.append(ivec % nside[i]) # Wrap indices for periodic boundary conditions

        # Normalize the Gaussian with the correct pseudocharge
        norm = pseudocharge[spe] / ((2.0*np.pi)**1.5 * sigma**3)
        
        # Build 3D Gaussian as the outer product of the 1D Gaussians along each axis, and add to the total density
        rho_n[np.ix_(idx[0], idx[1], idx[2])] += norm * gauss[0][:,None,None] * gauss[1][None,:,None] * gauss[2][None,None,:]

    return rho_n

def get_rho_n_rec(Gvec_half, knorm_vec, natoms, coords, atomic_symbols, pseudocharge, rloc):
    # Fourier transform of the Gaussian core charge distribution (DEPRECATED)
    rho_n_rec = np.zeros(len(Gvec_half), dtype=np.complex128)
    knorm2_vec = knorm_vec * knorm_vec
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        phase = np.exp(-1j * np.dot(Gvec_half, coords[iat]))
        rho_n_rec += pseudocharge[spe] * np.exp(-0.5 * knorm2_vec * rloc[spe]**2) * phase
    return rho_n_rec

def get_wn_real(cell, coords, atomic_symbols, nbasis, ncoefs, volume, pyscf_ewald, pyscf_core, rcut):
    # Short-range part of w_n = (Phi_i|rho_n)

    natoms = len(atomic_symbols)

    wn = np.zeros((ncoefs), dtype=np.float64)

    # Unpack PySCF data
    mol_bra = pyscf_ewald["mol_bra"]
    perm = pyscf_ewald["perm"]
    mol_core = pyscf_core["mol_core"]
    cslice = pyscf_core["coord_slice"]
    scale = pyscf_core["scale"]
    
    rcut_max = max(rcut.values()) # Pairs cutoff, to decided how many periodic images to include
    periodic = cell is not None and volume > 1.0e-10
    if periodic:
        cell = np.asarray(cell, dtype=float)
        inv_cell = np.linalg.inv(cell)
        images = lattice_images(cell, rcut_max, volume)
    else:
        images = np.zeros((1, 3))

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        pbra = perm[spe]
 
        block = np.zeros((nbasis[spe]), dtype=np.float64)
        for iat2 in range(natoms):
            spe2 = atomic_symbols[iat2]

            rcut_ij = rcut[(spe, spe2)] # real-space cutoff to use for this specific pair

            delta = coords[iat2] - coords[iat]

            if periodic:
                frac = delta @ inv_cell # Convert to fractional coordinates
                delta = (frac - np.round(frac)) @ cell # Wrap and convert back to Cartesian

            dvecs = delta + images
            keep = np.einsum("ij,ij->i", dvecs, dvecs) <= rcut_ij * rcut_ij # Keep only the images whose distance falls within the pair cutoff

            for d in dvecs[keep]:
                mol_core[spe2]._env[cslice[spe2]] = d # Move the core charge "ghost" atom
                raw = _pyscf_gto.intor_cross("int2c2e", mol_bra[spe], mol_core[spe2]) # Ask PySCF for the raw overlap integrals
                block += raw[pbra, 0] * scale[spe2] # Reorder PySCF's rows/columns into SALTED's (n,m) ordering and restore the core charge

        wn[icoefs:icoefs + nbasis[spe]] = block
        icoefs += nbasis[spe]

    return wn

def get_wn_rec(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs_ewald, rho_n_rec, volume, gcut):
    # Long-range part of w_n = (Phi_i|rho_n)

    wn = np.zeros((ncoefs), dtype=np.float64)

    Gvec_half = Gvec_half[:gcut]
    knorm2_vec = np.einsum('ij,ij->i', Gvec_half, Gvec_half)

    # Core charge density with Coulomb kernel
    rho_krnl = rho_n_rec[:gcut] / knorm2_vec
    rho_krnl_real = np.ascontiguousarray(rho_krnl.real)
    rho_krnl_imag = np.ascontiguousarray(rho_krnl.imag)

    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        phase = np.exp(-1j * np.dot(Gvec_half, coords[iat])) # e^{-iG.r_iat}
        obj = partial_wave_coefs_ewald[spe] * phase[:, np.newaxis]

        wn[icoefs:icoefs + nbasis[spe]] = (np.dot(obj.real.T, rho_krnl_real) + np.dot(obj.imag.T, rho_krnl_imag))
        icoefs += nbasis[spe]

    return wn * 2 * (4*np.pi)**2 / volume

def overlap_coulomb_rho(rho1_rec, rho2_rec, knorm_vec, volume):
    # Coulomb-metric overlap (rho1|rho2) in reciprocal space
    ovlp = np.sum((np.conj(rho1_rec)*rho2_rec).real * (4.0*np.pi) / knorm_vec**2)
    return 2 * ovlp / volume