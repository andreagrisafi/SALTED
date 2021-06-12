import sys,os
import numpy as np
import scipy.special as sc
import math
import time
import ctypes
from . import gausslegendre,nearfield_ewald,neighlist_ewald

def direct_ewald(sigewald,cell,nat,nnmax,nspecies,lmax,centers,all_species,nneightype,atom_indexes,rcut,coords,all_radial,sigma,sg,nmax,orthomatrix,radsize,lebsize):
    """return projections of the non-local field on basis functions"""

    start = time.time()
    alpha = 1.0/(2.0*sg**2)
    rcut += 4*sigewald
    rcut2 = rcut**2

    ncentype = len(centers)

    ncell = np.zeros(3,int)
    ncell[0] = int(np.round(rcut/np.linalg.norm(cell[:,0])))
    ncell[1] = int(np.round(rcut/np.linalg.norm(cell[:,1])))
    ncell[2] = int(np.round(rcut/np.linalg.norm(cell[:,2])))

    invcell = np.linalg.inv(cell)

    natmax = len(atom_indexes[0])

    # Do neighbour list
    nneigh_near,coordx_near = neighlist_ewald.neighlist_ewald(natmax,nat,nspecies,nnmax,coords.T,ncell,cell,invcell,rcut2,ncentype,np.array(centers,int),np.array(atom_indexes,int),np.array(all_species,int),np.array(nneightype,int))

    # Define atomic grid for potential 
    gauss_points,gauss_weights = gausslegendre.gauss_legendre.gaulegf(x1=0.0,x2=rcut*2.0,n=radsize)
    lebedev_grid = ld(lebsize)
    spherical_grid = np.zeros((lebsize*radsize,3),float)
    integration_weights = np.zeros((lebsize*radsize),float)
    igrid = 0
    for ir in range(radsize):
        r = gauss_points[ir]
        for ileb in range(lebsize):
            spherical_grid[igrid] = r * lebedev_grid[ileb][:3]
            integration_weights[igrid] = 4.0*np.pi * r**2 * lebedev_grid[ileb][3] * gauss_weights[ir] 
            igrid += 1

    # Get basis functions for potential projection 
    lr = np.sqrt(np.sum(spherical_grid**2,axis=1))
    lth = np.arccos(spherical_grid[:,2]/lr)
    lph = np.arctan2(spherical_grid[:,1],spherical_grid[:,0])

    harmonics = np.zeros(((lmax+1)**2,lebsize*radsize),complex) 
    lm = 0
    for l in range(lmax+1):
        for im in range(2*l+1):
            harmonics[lm,:] = np.conj(sc.sph_harm(im-l,l,lph[:],lth[:])) 
            lm += 1

    radial = radial_1D_mesh(sigma,nmax,lr,lebsize*radsize)
    orthoradial = np.dot(orthomatrix,radial)

    # compute near-field potential and project it onto the atomic basis
    omega_near = nearfield_ewald.nearfield(nat,nspecies,nmax,lmax,lebsize*radsize,nneigh_near,nnmax,alpha,coordx_near,spherical_grid,orthoradial,harmonics,integration_weights,sigewald) 
    omega_near = np.transpose(omega_near,(4,3,2,1,0))

    print("direct space potential computed in", time.time()-start, "seconds")
    print("")

    return omega_near



LIB = ctypes.CDLL(os.path.dirname(__file__) + "/lebedev.so")
LDNS = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
        590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890,
        4334, 4802, 5294, 5810]

def ld(n):
    '''Returns (x, y, z) coordinates along with weight w. Choose among [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810]'''

    if n not in LDNS:
        raise ValueError("n = {} not supported".format(n))
    xyzw = np.zeros((4, n), dtype=np.float64)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    n_out = ctypes.c_int(0)
    getattr(LIB, "ld{:04}_".format(n))(
        xyzw[0].ctypes.data_as(c_double_p),
        xyzw[1].ctypes.data_as(c_double_p),
        xyzw[2].ctypes.data_as(c_double_p),
        xyzw[3].ctypes.data_as(c_double_p),
        ctypes.byref(n_out))
    assert n == n_out.value
    return xyzw.T

def radial_1D_mesh(sigma, nmax, rvec, rsize):
    """Evaluate equispaced and normalized radial GTOs over a 1D mesh"""

    def rGTO(rval,nval,sigmaval):
        """Evaluate radial part of Gaussian type orbitals"""
        alphaval = 1.0/(2*(sigmaval)**2)
        f = rval**nval*np.exp(-alphaval*(rval)**2)
        return f

    # COMPUTE PRIMITIVE RADIAL FUNCTIONS
    radial = np.zeros((nmax,rsize),dtype=float)
    for n in range(nmax):
        inner = 0.5*sc.gamma(n+1.5)*(sigma[n]**2)**(n+1.5)
        radial[n,:] = rGTO(rvec[:],n,sigma[n])/np.sqrt(inner)

    return radial
