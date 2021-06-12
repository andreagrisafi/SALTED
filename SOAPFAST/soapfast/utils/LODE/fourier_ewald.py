import sys
import numpy as np
import scipy.special as sc
import math
import time
from . import phasecomb,gcontra,fourier_integrals

def fourier_ewald(sigewald,nat,nnmax,nspecies,lmax,centers,all_species,nneighmax,atom_indexes,cell,rcut,coords,all_radial,sigma,sg,nmax,orthomatrix,nside,iGx,imGx,Gval,Gvec,nG):
    """return projections of the non-local field on basis functions"""

    volume = np.linalg.det(cell)

    start = time.time()
    alpha = 1.0/(2.0*sg**2)

    start1 = time.time()
    # process coordinates 
    coordx = np.zeros((nat,nspecies,nat,3), dtype=float)
    nneigh = np.zeros((nat,nspecies),int)
    iat = 0
    ncentype = len(centers)
    # loop over species to center on
    for icentype in range(ncentype):
        centype = centers[icentype]
        # loop over centers of that species
        for icen in range(nneighmax[centype]):
            cen = atom_indexes[centype,icen]
            # loop over all the species to use as neighbours
            for ispe in range(nspecies):
                spe = all_species[ispe]
                # loop over neighbours of that species
                n = 0
                for ineigh in range(nneighmax[spe]):
                    neigh = atom_indexes[spe,ineigh]
                    coordx[iat,ispe,n,0] = coords[neigh,0] - coords[cen,0]
                    coordx[iat,ispe,n,1] = coords[neigh,1] - coords[cen,1]
                    coordx[iat,ispe,n,2] = coords[neigh,2] - coords[cen,2]
                    nneigh[iat,ispe] += 1
                    n += 1
            iat = iat + 1

    start2 = time.time()
    # combine phase factors
    phase = phasecomb.phasecomb(nat,nspecies,nneigh,nG,coordx,Gvec.T)
#    print "phases combinations :", time.time()-start2, "seconds"

    start3 = time.time()
    # compute analytic radial integrals and spherical harmonics
    alphaewald = 1.0/(2.0*sigewald**2)
    [orthoradint,harmonics] = fourier_integrals.fourier_integrals(nG,nmax,lmax,alphaewald,rcut,sigma,Gval,Gvec,orthomatrix) 
    orthoradint = np.moveaxis(orthoradint, 0, -1)
#    print "fourier integrals :", time.time()-start3, "seconds"

    start4 = time.time()
    # perform contraction over G-vectors
    omega = gcontra.gcontra(nat,nspecies,nmax,lmax,nG,orthoradint,harmonics.T,2.0*phase)
#    print "G-contraction :", time.time()-start4, "seconds"

#    print "-----------------------------------------"
    print("reciprocal space potential computed in", time.time()-start, "seconds")

    omega *= 16.0*np.pi**2/volume      

    return omega
