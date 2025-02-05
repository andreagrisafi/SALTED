import os
import sys
import time
import os.path as osp
import numpy as np
from scipy import special
from salted.sph_utils import complex_to_real_transformation

def init_moments(inp,species,lmax,nmax,rank):
    """Compute basis function integrals relevant for computing total charge, dipole and polarizability tensor"""

    if rank==0:

        if inp.salted.saltedtype=="density":

            print("Total charges and polarization vectors will be computed from the reference and predicted electron densities.")
            print("WARNING: Computed values of polarization vectors have physical meaning only along those Cartesian directions for which the electron density goes to zero at the cell periodic boundaries. The modern theory of polarization should be used otherwise.")

        elif inp.salted.saltedtype=="density-response":
            
            print("Polarizability tensors will be computed from the reference and predicted density-response functions.")
            print("WARNING: Computed values of polarizability tensors have physical meaning only along those Cartesian directions for which the electron density goes to zero at the cell periodic boundaries. The modern theory of polarization should be used otherwise.")

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

def compute_charge_and_dipole(geom,pseudocharge,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,dipole_integrals,coefs,average):
    """Compute total charge and dipole moment for the given configuration"""

    geom.wrap()
    bohr2angs = 0.529177210670
    coords = geom.get_positions()/bohr2angs
    all_symbols = geom.get_chemical_symbols()
    all_natoms = int(len(all_symbols))

    # Compute unnormalized electron-density integral
    iaux = 0
    nele = 0.0
    charge = 0.0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        nele += pseudocharge
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                if l==0:
                    charge += charge_integrals[(spe,l,n)] * coefs[iaux]
                iaux += 2*l+1

    # Initialize dipole 
    cart = ["y","z","x"]
    dipole = {}
    for icart in range(3):
        dipole[cart[icart]] = 0.0
    
    # Perform dipole calculation
    iaux = 0
    for iat in range(all_natoms):
        spe = all_symbols[iat]
        if spe in species:
            if average:
                # Add contribution of nuclear pseudocharge to the dipole
                dipole["x"] += pseudocharge * coords[iat,0] 
                dipole["y"] += pseudocharge * coords[iat,1] 
                dipole["z"] += pseudocharge * coords[iat,2] 
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0:
                            if average:
                                # rescale isotropic coefficients to conserve the electronic charge
                                coefs[iaux] *= nele/charge
                            else:
                                # remove residual charge from the most diffuse isotropic function
                                if n==nmax[(spe,l)]-1:
                                    coefs[iaux] -= charge/(charge_integrals[(spe,l,n)]*natoms)
                            # Compute l=0 electronic contribution to the dipole 
                            # NB: this is ill-defined in a truly periodic system and/or for systems with a net charge
                            dipole["x"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,0]
                            dipole["y"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,1]
                            dipole["z"] -= coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                        if l==1:
                            # Compute l=1 electronic contribution to the dipole 
                            # NB: this follows the correspondence between (-1,0,1) real spherical harmonics and (y,z,x) Cartesian coordinates 
                            dipole[cart[im]] -= coefs[iaux] * dipole_integrals[(spe,l,n)]
                        iaux += 1

    return [charge,dipole]

def compute_polarizability(geom,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,dipole_integrals,coefs):
    """Compute polarizability tensor for the given configuration"""

    geom.wrap()
    bohr2angs = 0.529177210670
    coords = geom.get_positions()/bohr2angs
    all_symbols = geom.get_chemical_symbols()
    all_natoms = int(len(all_symbols))

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
        for iat in range(all_natoms):
            spe = all_symbols[iat]
            if spe in species:
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

def init_ghost_integrals(inp,cell,lmax,nmax,species):
    """Compute the geometric integrals required to compute the center of charge of a 3D-periodic ghost electron-density following the topological definition derived from the modern theory of polarization. A fixed orthorombic cell is assumed."""

    kmin = np.zeros((3,3))
    kmin[0] = np.array([2*np.pi/cell[0,0],0.0,0.0])
    kmin[1] = np.array([0.0,2*np.pi/cell[1,1],0.0])
    kmin[2] = np.array([0.0,0.0,2*np.pi/cell[2,2]])

    # Get CP2K basis set information 
    bdir = osp.join(inp.salted.saltedpath,"basis")
    alphas = {}
    contra = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            alphas[(spe,l)] = np.atleast_1d(np.loadtxt(osp.join(bdir,f"{spe}-{inp.qm.dfbasis}-alphas-L{l}.dat")))
            contra[(spe,l)] = np.atleast_2d(np.loadtxt(osp.join(bdir,f"{spe}-{inp.qm.dfbasis}-contra-L{l}.dat")))

    # Compute basis function integrals 
    kmin_integrals = {}
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
                # Loop over the 3 Cartesian directions
                for ik in range(3):
                    knorm = np.linalg.norm(kmin[ik])
                    kmin_radint = 0.0 
                    # Perform contraction over primitive GTOs
                    for ipgf in range(npgf):
                        # Compute Gaussian sigma 
                        sigma = np.sqrt(0.5/alphas[(spe,l)][ipgf])
                        # Compute \int_0^\infty dr r^2 * r^{l} * \exp[-r^2/(2\sigma^2)] * j_l(k_min*r) 
                        kmin_radint += contra[(spe,l)][n,ipgf] * knorm**l * np.exp(-0.5*(knorm*sigma)**2) * np.sqrt(np.pi/2.0) * sigma**(3+2*l)
                    # Normalize
                    kmin_integrals[(ik,spe,l,n)] = kmin_radint / np.sqrt(inner)

    kmin_harmonics = {}
    for ik in range(3):
        theta = np.arccos(kmin[ik,2]/np.linalg.norm(kmin[ik]))
        phi = np.arctan2(kmin[ik,1],kmin[ik,0])
        for spe in species:
            for l in range(lmax[spe]+1):
                kmin_harmonics[(ik,spe,l)] = np.zeros(2*l+1,complex)
                for im in range(2*l+1):
                    m = im-l
                    kmin_harmonics[(ik,spe,l)][im] = special.sph_harm(m, l, phi, theta)

    return [kmin_integrals,kmin_harmonics]

def compute_ghost_center(geom,natoms,atomic_symbols,lmax,nmax,species,charge_integrals,kmin_integrals,kmin_harmonics,coefs):
    """Compute the center of charge of a 3D-periodic ghost electron-density following the topological definition derived from the modern theory of polarization."""

    geom.wrap()
    bohr2angs = 0.529177210670
    coords = geom.get_positions()/bohr2angs
    cell = geom.get_cell()/bohr2angs
    all_symbols = geom.get_chemical_symbols()
    all_natoms = int(len(all_symbols))

    # Compute total charge 
    charge = 0.0
    iaux = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                if l==0:
                    charge += charge_integrals[(spe,l,n)] * coefs[iaux]
                iaux += 2*l+1
    print("Charge before normalization = ", charge)   
 
    kmin = np.zeros((3,3))
    kmin[0] = np.array([2*np.pi/cell[0,0],0.0,0.0])
    kmin[1] = np.array([0.0,2*np.pi/cell[1,1],0.0])
    kmin[2] = np.array([0.0,0.0,2*np.pi/cell[2,2]])

    # Compute center of charge
    center_of_charge = np.zeros(3)
    for ik in range(3):
        iaux = 0
        fourier_coef = 0.0+0.0j 
        for iat in range(all_natoms):
            phase_factor = np.exp(- 1.0j * np.dot(kmin[ik],coords[iat]))
            spe = all_symbols[iat]
            if spe in species:
                for l in range(lmax[spe]+1):
                    c2r = complex_to_real_transformation([2*l+1])[0]
                    for n in range(nmax[(spe,l)]):
                        if l==0:
                           # Rescale coefficients to ensure total charge integrates to 1 electron
                           coefs[iaux] *= 1.0/charge
                        # Make coefficients complex and take the complex conjugate: c_real --> c*
                        coefs_complex = np.array(np.dot(c2r.T,coefs[iaux:iaux+2*l+1]),complex)
                        # Collect contributions to fourier coefficients
                        for im in range(2*l+1):
                            fourier_coef += (-1.0j)**l * phase_factor * coefs_complex[im] * kmin_integrals[(ik,spe,l,n)] * kmin_harmonics[(ik,spe,l)][im]
                            iaux += 1
        fourier_coef *= 4*np.pi
        berry_phase = np.imag(np.log(fourier_coef))
        center_of_charge[ik] = berry_phase * (-1.0/kmin[ik,ik])
        
    return center_of_charge 
