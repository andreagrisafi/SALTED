cimport cython # the main Cython stuff

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel cimport prange

cimport scipy.special.cython_special as csc # Cython interfaces to Scipy Special functions (sph_harm, gamma, hyp1f1)

from libc.math cimport sqrt,abs
from libc.stdio cimport printf

cdef extern void zero_array (double *array, double value, size_t n)

# NumPy and Scipy Special Functions
import sys
import numpy as np
import scipy.special as sc

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _radialscaling(double r, double c, double r0, double m):

    if (m == 0):
        return 1.0
    if (c == 0.0):
        return 1.0 / (r/r0)**m
    else:
        return c / (c + (r/r0)**m)

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _initsoapmolecule(long nspecies,
                           long lmax,
                           long[:] centers,
                           long[:] all_species,
                           long[:] nneighmax,
                           long[:,:] atom_indexes,
                           double rcut,
                           double alpha,
                           double[:,:] coords,
                           long[:,:] nneigh,
                           double[:,:,:] length,
                           double[:,:,:] efact,
                           complex[:,:,:,:,:] harmonic,
                           double radial_c,
                           double radial_r0,
                           double radial_m
                    ):

    # Py_ssize_t is the correct C type for Python array indexes
    cdef Py_ssize_t icentype,centype,icen,cen,ispe,ineigh,neigh,lval,mval,im,n 
    cdef double rx,ry,rz,ph,th,r2,rdist
 
    iat = 0
    ncentype = len(centers)
    # loop over species to center on
    for icentype in xrange(ncentype):
        centype = centers[icentype]
        # loop over centers of that species
        for icen in xrange(nneighmax[centype]):
            cen = atom_indexes[centype,icen]
            # loop over all the species to use as neighbours
            for ispe in xrange(nspecies):
                spe = all_species[ispe]
                # loop over neighbours of that species
                n = 0
                for ineigh in xrange(nneighmax[spe]):
                    neigh = atom_indexes[spe,ineigh]
                    # compute distance vector
                    rx = coords[neigh,0] - coords[cen,0] 
                    ry = coords[neigh,1] - coords[cen,1] 
                    rz = coords[neigh,2] - coords[cen,2]
                    r2 = rx**2 + ry**2 + rz**2
                    # within cutoff ?
                    if r2 <= rcut**2:
                        # central atom ?
                        if neigh == cen:
                            length[iat,ispe,n]  = 0.0
                            efact[iat,ispe,n]   = 1.0 
                            harmonic[iat,ispe,0,0,n] = sc.sph_harm(0,0,0,0)
                            nneigh[iat,ispe] = nneigh[iat,ispe] + 1
                            n = n + 1
                        else:
                            rdist = np.sqrt(r2)
                            length[iat,ispe,n]  = rdist
                            th = np.arccos(rz/rdist)
                            ph = np.arctan2(ry,rx)
                            efact[iat,ispe,n] = np.exp(-alpha*r2) * _radialscaling(rdist,radial_c,radial_r0,radial_m)
                            for lval in xrange(lmax+1):
                                for im in xrange(2*lval+1):
                                    mval = im-lval
                                    harmonic[iat,ispe,lval,im,n] = np.conj(sc.sph_harm(mval,lval,ph,th)) 
                            nneigh[iat,ispe] = nneigh[iat,ispe] + 1
                            n = n + 1
            iat = iat + 1

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _initsoapperiodic(long nspecies,
                           long[:] ncell,
                           long lmax,
                           long[:] centers,
                           long[:] all_species,
                           long[:] nneighmax,
                           long[:,:] atom_indexes,
                           double rcut,
                           double alpha,
                           double[:,:] coords,
                           double[:,:] cell,
                           double[:,:] invcell,
                           long[:,:] nneigh,
                           double[:,:,:] length,
                           double[:,:,:] efact,
                           complex[:,:,:,:,:] harmonic,
                           double radial_c,
                           double radial_r0,
                           double radial_m
                    ):

    # Py_ssize_t is the correct C type for Python array indexes
    cdef Py_ssize_t icentype,centype,icen,cen,ispe,ineigh,neigh,ia,ib,ic,lval,mval,im,n 
    cdef double rx,ry,rz,rcx,rcy,rcz,sx,sy,sz,ph,th,r2,rdist,rrx,rry,rrz

    iat = 0
    ncentype = len(centers)
    # loop over species to center on
    for icentype in xrange(ncentype):
        centype = centers[icentype]
        # loop over centers of that species
        for icen in xrange(nneighmax[centype]):
            cen = atom_indexes[centype,icen]
            # loop over all the species to use as neighbours
            for ispe in xrange(nspecies):
                spe = all_species[ispe]
                # loop over neighbours of that species
                n = 0
                for ineigh in xrange(nneighmax[spe]):
                    neigh = atom_indexes[spe,ineigh]
                    # compute distance vector
                    rx = coords[neigh,0] - coords[cen,0] 
                    ry = coords[neigh,1] - coords[cen,1] 
                    rz = coords[neigh,2] - coords[cen,2]
                    # apply pbc 
                    sx = invcell[0,0]*rx + invcell[0,1]*ry + invcell[0,2]*rz
                    sy = invcell[1,0]*rx + invcell[1,1]*ry + invcell[1,2]*rz
                    sz = invcell[2,0]*rx + invcell[2,1]*ry + invcell[2,2]*rz
                    sx = sx - np.round(sx)
                    sy = sy - np.round(sy)
                    sz = sz - np.round(sz)
                    rcx = cell[0,0]*sx + cell[0,1]*sy + cell[0,2]*sz
                    rcy = cell[1,0]*sx + cell[1,1]*sy + cell[1,2]*sz
                    rcz = cell[2,0]*sx + cell[2,1]*sy + cell[2,2]*sz
                    # replicate cell
                    for ia in xrange(-ncell[0],ncell[0]+1):
                        for ib in xrange(-ncell[1],ncell[1]+1):
                            for ic in xrange(-ncell[2],ncell[2]+1):
                                rrx = rcx + ia*cell[0,0] + ib*cell[0,1] + ic*cell[0,2]
                                rry = rcy + ia*cell[1,0] + ib*cell[1,1] + ic*cell[1,2]
                                rrz = rcz + ia*cell[2,0] + ib*cell[2,1] + ic*cell[2,2]
                                r2 = rrx**2 + rry**2 + rrz**2
                                # within cutoff ?
                                if r2 <= rcut**2:
                                    # central atom ?
                                    if neigh == cen and ia==0 and ib==0 and ic==0:
                                        length[iat,ispe,n]  = 0.0
                                        efact[iat,ispe,n]   = 1.0 
                                        harmonic[iat,ispe,0,0,n] = sc.sph_harm(0,0,0,0)
                                        nneigh[iat,ispe] = nneigh[iat,ispe] + 1
                                        n = n + 1
                                    else:
                                        rdist = np.sqrt(r2)
                                        length[iat,ispe,n]  = rdist
                                        th = np.arccos(rrz/rdist)
                                        ph = np.arctan2(rry,rrx)
                                        efact[iat,ispe,n] = np.exp(-alpha*r2) * _radialscaling(rdist,radial_c,radial_r0,radial_m)
                                        for lval in xrange(lmax+1):
                                            for im in xrange(2*lval+1):
                                                mval = im-lval
                                                harmonic[iat,ispe,lval,im,n] = np.conj(sc.sph_harm(mval,lval,ph,th)) 
                                        nneigh[iat,ispe] = nneigh[iat,ispe] + 1
                                        n = n + 1
            iat = iat + 1

#----------------------------------------------------------------------------------------------------------------------------------------

def initsoap(nat,nnmax,nspecies,lmax,centers,all_species,nneighmax,atom_indexes,rcut,coords,cell,all_radial,sigma,sg,nmax,orthomatrix):
    """return initialization variables for SOAP"""

    alpha = 1.0 / (2.0 * sg**2)
    sg2 = sg**2

    nneigh      = np.zeros((nat,nspecies), dtype=int)
    length      = np.zeros((nat,nspecies,nnmax), dtype=float)
    efact       = np.zeros((nat,nspecies,nnmax), dtype=float)
    omega       = np.zeros((nat,nspecies,nmax,lmax+1,2*lmax+1),complex)
    harmonic    = np.zeros((nat,nspecies,lmax+1,2*lmax+1,nnmax), dtype=complex)
    radint      = np.zeros((nat,nspecies,nnmax,lmax+1,nmax),float)
    orthoradint = np.zeros((nat,nspecies,lmax+1,nmax,nnmax),float)

    if (np.sum(cell) == 0.0):
        _initsoapmolecule(nspecies,lmax,centers,all_species,nneighmax,atom_indexes,rcut,alpha,coords,nneigh,length,efact,harmonic,all_radial[0],all_radial[1],all_radial[2])
    else:
        ncell = np.zeros(3,int)
        ncell[0] = int(np.round(rcut/np.linalg.norm(cell[:,0])))
        ncell[1] = int(np.round(rcut/np.linalg.norm(cell[:,1])))
        ncell[2] = int(np.round(rcut/np.linalg.norm(cell[:,2])))

        invcell = np.linalg.inv(cell)
        _initsoapperiodic(nspecies,ncell,lmax,centers,all_species,nneighmax,atom_indexes,rcut,alpha,coords,cell,invcell,nneigh,length,efact,harmonic,all_radial[0],all_radial[1],all_radial[2])

    for n in xrange(nmax):
        normfact = np.sqrt(2.0/(sc.gamma(1.5+n)*sigma[n]**(3.0+2.0*n)))
        sigmafact = (sg2**2+sg2*sigma[n]**2)/sigma[n]**2
        for l in xrange(lmax+1):
            radint[:,:,:,l,n] = efact[:,:,:] \
                                * 2.0**(-0.5*(1.0+l-n)) \
                                * (1.0/sg2 + 1.0/sigma[n]**2)**(-0.5*(3.0+l+n)) \
                                * sc.gamma(0.5*(3.0+l+n))/sc.gamma(1.5+l) \
                                * (length[:,:,:]/sg2)**l \
                                * sc.hyp1f1(0.5*(3.0+l+n), 1.5+l, 0.5*length[:,:,:]**2/sigmafact)
        radint[:,:,:,:,n] *= normfact

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            for neigh in xrange(nneigh[iat,ispe]):
                for l in xrange(lmax+1):
                    orthoradint[iat,ispe,l,:,neigh] = np.dot(orthomatrix,radint[iat,ispe,neigh,l])

    for iat in xrange(nat):
        for ispe in xrange(nspecies):
            omega[iat,ispe] = np.einsum('lnh,lmh->nlm',orthoradint[iat,ispe],harmonic[iat,ispe])

    return [omega,harmonic,orthoradint]
