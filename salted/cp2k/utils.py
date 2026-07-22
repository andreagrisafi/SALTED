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


   # Dict with key as strings and values of type float array
   partial_wave_coefs = Dict.empty(key_type=types.unicode_type,value_type=types.complex128[:,:])
   for spe in species:
      partial_wave_coefs[spe] = np.zeros((nG_loc, nbasis[spe]), dtype=np.complex128)

   for iG in prange(nG_loc):

      kx = Gvec[iG,0]
      ky = Gvec[iG,1]
      kz = Gvec[iG,2]

      # Norm square of the k-mode vector
      knorm2 = kx*kx + ky*ky + kz*kz


      for spe in species:

         knorm = np.sqrt(knorm2)
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

            harmonics = np.zeros((2*lmax[spe]+1))
            radintk = np.zeros((max(nmax.values())))

            lamfactor = np.sqrt(np.pi/2.0) * knorm**lam

            # compute orthonormalized real spherical harmonics with Condon-Shortley phase convention
            for mu in range(2*lam+1):
               harmonics[mu] = spherical_harmonic(lam,mu-lam,costheta,phi)
               
            pradintk = np.zeros((npgf[key]))

            for ipgf in range(npgf[key]):
               # compute normalized radial integral
               sigma = np.sqrt(1.0 / (2.0 * alphas[key][ipgf]))
               pradintk[ipgf] = lamfactor * sigma**(2.0*lam+3.0) * np.exp(-0.5*knorm2*(sigma**2))

            # Precompute partial wave coefficients <nlm|k> consisting in
            # spherical harmonics and radial integrals evaluated at the given k
            for irad in range(nmax[key]):
               radintk[irad] = 0.0
               for ipgf in range(npgf[key]):
                  radintk[irad] += contranorm[key][irad,ipgf]*pradintk[ipgf]
               for mu in range(2*lam+1):
                  partial_wave_coefs[spe][iG,ibasis] = radintk[irad] * harmonics[mu] * (( -1.0j)**(np.float64(lam)))
                  ibasis = ibasis + 1

   return partial_wave_coefs

@njit(parallel = True, fastmath = True)
def gto_rec_prim(lmax, species, npgf, alphas, Gvec, nG_loc):
    # Fourier transform of primitive atom-centered basis functions
    
    partial_wave_coefs = Dict.empty(key_type=types.unicode_type, value_type=types.complex128[:, :, :]) # Dict with key as strings and values of type float array
    
    for spe in species:
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            # npgf[key]: number of individual Gaussians, each with its own alpha
            nmu = 2*lam + 1  #number of m values (-lam,...,+lam) for this lam
            partial_wave_coefs[key] = np.zeros((nG_loc, npgf[key], nmu), dtype=np.complex128)

    for iG in prange(nG_loc):

        kx = Gvec[iG, 0]
        ky = Gvec[iG, 1]
        kz = Gvec[iG, 2]

        # Norm squared |G|^2 and norm |G| of the k-mode vector
        knorm2 = kx*kx + ky*ky + kz*kz
        knorm = np.sqrt(knorm2)
        
        # Direction of G in spherical angles (costheta, phi)
        if knorm == 0.0:
            costheta = 0.0
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
                    sigma2 = 1.0 / (2.0 * alphas[key][ipgf]) # Squared Gaussian width in reciprocal space
                    sigma = np.sqrt(sigma2) # Gaussian width in reciprocal space
                    
                    # Radial integral
                    radial = lamfactor * sigma2**lam * sigma**3.0 * np.exp(-0.5*knorm2*sigma2)
                    for mu in range(2*lam+1):
                        partial_wave_coefs[key][iG, ipgf, mu] = radial * harmonics[mu] * phase_lam

    return partial_wave_coefs

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
def build_matrices_prim(Gvec_half, natoms, coords, npgf, lmax, atomic_symbols, partial_wave_coefs, rho_KS_rec, df_metric, ncut, rank):
    # Build the primitive-basis overlap matrix Sp and density vector wp
    
    # Get the total size of the primitive basis
    ncoefs_prim = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            ncoefs_prim += npgf[key] * (2*lam + 1)

    Sp = np.zeros((ncoefs_prim, ncoefs_prim), dtype=np.float64)
    wp = np.zeros((ncoefs_prim), dtype=np.float64)

    knorm_vec = np.sqrt(np.sum(Gvec_half*Gvec_half,axis=1)).astype(np.float64) # |G|
    phase = np.exp(-1j * np.dot(Gvec_half, coords.T)) # e^{-iG.r_iat}
    if df_metric == "coulomb":
        phase_over_knorm = phase / knorm_vec[:, np.newaxis] 

    # Outer loop over (iat, lam, mu) indexes rows of Sp and entries of wp.
    ipgf = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            
            # wp calculation
            for mu in range(2*lam + 1):

                obj1_list = []
                for ipgf1 in range(npgf[key]):
                    n1 = ncut[key][ipgf1]
                    if df_metric == "identity":
                        obj1 = phase[:n1, iat] * partial_wave_coefs[key][:n1, ipgf1, mu]
                        wp[ipgf+ipgf1] = 2 * (np.dot(obj1[1:].real, rho_KS_rec[1:n1].real) + np.dot(obj1[1:].imag, rho_KS_rec[1:n1].imag))
                        wp[ipgf+ipgf1] += obj1[0].real * rho_KS_rec[0].real + obj1[0].imag * rho_KS_rec[0].imag
                    if df_metric == "coulomb":
                        obj1 = phase_over_knorm[:n1, iat] * partial_wave_coefs[key][:n1, ipgf1, mu]
                        wp[ipgf+ipgf1] = 2 * (np.dot(obj1.real, rho_KS_rec[:n1].real / knorm_vec[:n1]) + np.dot(obj1.imag, rho_KS_rec[:n1].imag / knorm_vec[:n1]))
                    obj1_list.append(obj1)

                # Sp calculation
                # Inner loop over (iat2, lam2, mu2) indexes columns of Sp
                ipgf2 = 0
                for iat2 in range(iat+1):
                    spe2 = atomic_symbols[iat2]
                    for lam2 in range(lmax[spe2]+1):
                        key2 = f"{spe2}_{lam2}"
                        for mu2 in range(2*lam2 + 1):
                            for ipgf2_local in range(npgf[key2]):
                                n2 = ncut[key2][ipgf2_local]
                                
                                if df_metric == "identity":
                                    obj2 = phase[:n2, iat2] * partial_wave_coefs[key2][:n2, ipgf2_local, mu2]
                                if df_metric == "coulomb":
                                    obj2 = phase_over_knorm[:n2, iat2] * partial_wave_coefs[key2][:n2, ipgf2_local, mu2]
                                    
                                for ipgf1 in range(npgf[key]):
                                    obj1 = obj1_list[ipgf1]
                                    n1 = ncut[key][ipgf1]
                                    ncut_pair = min(n1, n2)
 
                                    if df_metric == "identity":
                                        val = 2 * (np.dot(obj1[1:ncut_pair].real, obj2[1:ncut_pair].real) + np.dot(obj1[1:ncut_pair].imag, obj2[1:ncut_pair].imag))
                                        val += obj1[0].real*obj2[0].real + obj1[0].imag*obj2[0].imag
                                    if df_metric == "coulomb":
                                        val = 2 * (np.dot(obj1[:ncut_pair].real, obj2[:ncut_pair].real) + np.dot(obj1[:ncut_pair].imag, obj2[:ncut_pair].imag))
 
                                    Sp[ipgf+ipgf1, ipgf2+ipgf2_local] = val
                                    if ipgf+ipgf1 != ipgf2+ipgf2_local:
                                        Sp[ipgf2+ipgf2_local, ipgf+ipgf1] = val

                            ipgf2 += npgf[key2]
                ipgf += npgf[key]

    Sp = np.real(Sp)*4*np.pi
    wp = np.real(wp)

    return Sp, wp

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

def build_ncutoff(alphas, npgf, species, lmax, knorm_vec, nG_half):
    ncut = Dict.empty(key_type=types.unicode_type, value_type=types.int64[:])
    for spe in species:
        for lam in range(lmax[spe]+1):
            key = f"{spe}_{lam}"
            gmax = gmax_for_prim(alphas[key])
            ncut[key] = np.searchsorted(knorm_vec, gmax).astype(np.int64) #Find index where Gmax would be inserted to to maintain order.
            #ncut[key] = np.full(npgf[key], nG_half, dtype=np.int64) # For debugging purposes, set ncut to the full size of G.
    return ncut