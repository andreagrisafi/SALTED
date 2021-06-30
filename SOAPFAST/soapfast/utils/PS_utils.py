#!/usr/bin/python

import sys
import numpy as np
from sympy.physics.wigner import wigner_3j
from scipy import special
from . import initsoap

#############################################################################################

def setup_orthomatrix(nmax,rc):

    sigma = np.zeros(nmax,float)
    for i in range(nmax):
        sigma[i] = max(np.sqrt(float(i)),1.0)*(rc)/float(nmax)

    overlap = np.zeros((nmax,nmax),float)
    for n1 in range(nmax):
        for n2 in range(nmax):
            overlap[n1,n2] = (0.5/(sigma[n1])**2 + 0.5/(sigma[n2])**2)**(-0.5*(3.0 +n1 +n2)) \
                             /(sigma[n1]**n1 * sigma[n2]**n2)*\
                              special.gamma(0.5*(3.0 + n1 + n2))/ (  \
                    (sigma[n1]*sigma[n2])**1.5 * np.sqrt(special.gamma(1.5+n1)*special.gamma(1.5+n2)) )    

    eigenvalues, unitary = np.linalg.eig(overlap)
    sqrteigen = np.sqrt(eigenvalues) 
    diagoverlap = np.diag(sqrteigen)
    newoverlap = np.dot(np.conj(unitary),np.dot(diagoverlap,unitary.T))
    orthomatrix = np.linalg.inv(newoverlap)

    return [orthomatrix,sigma]

##################################################################################################################

def compute_power_spectrum(nat,nneighmax,natmax,lam,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,atom_indexes,all_species,coords,cell,rc,cw,sigma,sg,orthomatrix,sparse_options,all_radial,ncen,useall,verbose):


    # Get number of features
    if lam == 0:
        featsize = nspecies*nspecies*nmax**2*(lmax+1)
    else:
        featsize = nspecies*nspecies*nmax**2*llmax
        # Precompute Wigner 3j symbols
        w3j = np.zeros((2*lam+1,lmax+1,lmax+1,2*lmax+1),float)
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                for m in range(2*l1+1):
                    for mu in range(2*lam+1):
                        w3j[mu,l1,l2,m] = wigner_3j(lam,l2,l1,mu-lam,m-l1-mu+lam,-m+l1) * (-1.0)**(m-l1)

    # Get number of retained features
    if sparse_options[0] != '':
        do_list = sparse_options[1]
    else:
        do_list = range(featsize)

    # Allocate arrays
    if lam == 0:
        PS = np.zeros((npoints,natmax,len(do_list)),dtype=complex)
    else:
        PS = np.zeros((npoints,natmax,2*lam+1,len(do_list)),dtype=complex)

    # Padding for verbose mode:
    npad = len(str(npoints))

    if lam == 0:

        # Find out which components we want to keep.
        keep_components = np.zeros((nspecies,nspecies,nmax,nmax,lmax+1),dtype=object)
        for i in range(nspecies):
            for j in range(nspecies):
                for k in range(nmax):
                    for l in range(nmax):
                        for m in range(lmax+1):
                            keep_components[i,j,k,l,m] = [i,j,k,l,m]
 

        keep_components = keep_components.reshape(featsize)
        if (sparse_options[0] != ''):
            keep_components = keep_components[do_list]
            comp = []
            for component in keep_components:
                comp.append([component[0],component[2],component[4]])
                comp.append([component[1],component[3],component[4]])
            all_keep_components = comp
        else:
            keep_components = []
        if useall:
            keep_components = []

        for i in range(npoints):

            # compute the first kind of orthogonal projections from the SOAP density 
            [omega1,harmonic1,orthoradint1] = initsoap.initsoap(nat[i],nnmax,nspecies,lmax,centers,all_species,nneighmax[i],atom_indexes[i],rc,coords[i],cell[i].T,all_radial,sigma,sg,nmax,orthomatrix)


            omega2 = omega1.copy()



            if verbose:
                strg = "Doing point %*i of %*i (%6.2f %%)"%(npad,i+1,npad,npoints,100 * float(i+1)/npoints)
                sys.stdout.write('%s\r'%strg)
                sys.stdout.flush()

            # Precompute omega conjugate
            omegatrue = np.zeros((nat[i],nspecies,nmax,lmax+1,2*lmax+1),complex)
            omegaconj = np.zeros((nat[i],nspecies,nmax,lmax+1,2*lmax+1),complex)
            for l in range(lmax+1):
                for im in range(2*l+1):
                    omegatrue[:,:,:,l,im] = omega2[:,:,:,l,im]/np.sqrt(np.sqrt(2*l+1))
                    omegaconj[:,:,:,l,im] = np.conj(omega1[:,:,:,l,im])/np.sqrt(np.sqrt(2*l+1))
    

            # Compute power spectrum
            if (len(keep_components)==0):

                power = np.einsum('asblm,adnlm->asdbnl',omegatrue,omegaconj)
                power = power.reshape(nat[i],featsize)
                PS[i,:nat[i]] = power[:,do_list]

            else:

                featsize2 = len(do_list)
                power = np.zeros((nat[i],featsize2),dtype=complex)
                for cc in range(len(keep_components)):
                    comp = keep_components[cc]
                    for iat in range(nat[i]):
                        power[iat,cc] = np.dot(omegatrue[iat,comp[0],comp[2],comp[4],:],omegaconj[iat,comp[1],comp[3],comp[4],:])
                PS[i,:nat[i]] = power[:]

        
    else:


        # Find out which components we want to keep.
        keep_components = np.zeros((nspecies,nspecies,nmax,nmax,llmax),dtype=object)
        for i in range(nspecies):
            for j in range(nspecies):
                for k in range(nmax):
                    for l in range(nmax):
                        for m in range(llmax):
                            keep_components[i,j,k,l,m] = [i,j,k,l,lvalues[m][0],lvalues[m][1]]
        keep_components = keep_components.reshape(nspecies*nspecies*nmax*nmax*llmax)


        if (sparse_options[0] != ''):
            keep_components = keep_components[do_list]
        else:
            keep_components = []
        if useall:
            keep_components = []
      
        # Compute harmonic conjugate, omega conjugate and tensorial power spectrum
        # reduce l dimensionality keeping the nonzero elements
        for i in range(npoints):

            # compute the first kind of orthogonal projections from the SOAP density 
            [omega1,harmonic1,orthoradint1] = initsoap.initsoap(nat[i],nnmax,nspecies,lmax,centers,all_species,nneighmax[i],atom_indexes[i],rc,coords[i],cell[i].T,all_radial,sigma,sg,nmax,orthomatrix)


            omega2 = omega1.copy()


            if verbose:
                strg = "Doing point %*i of %*i (%6.2f %%)"%(npad,i+1,npad,npoints,100 * float(i+1)/npoints)
                sys.stdout.write('%s\r'%strg)
                sys.stdout.flush()

            if (len(keep_components)==0):

                # Fill power spectrum arrays
                omegaconj = np.zeros((nat[i],nspecies,2*lam+1,nmax,lmax+1,lmax+1,2*lmax+1),complex)
                harmconj = np.zeros((nat[i],nspecies,lmax+1,lmax+1,2*lmax+1,2*lam+1,nnmax),dtype=complex)
    
                for lval in range(lmax+1):
                    for im in range(2*lval+1):
                        for lval2 in range(lmax+1):
                            for mu in range(2*lam+1):
                                if abs(im-lval-mu+lam) <= lval2:
                                    harmconj[:,:,lval2,lval,im,mu,:] = np.conj(harmonic1[:,:,lval2,lval2+im-lval-mu+lam,:])
                for iat in range(ncen[i]):
                    for ispe in range(nspecies): 
                        omegaconj[iat,ispe] = np.einsum('lnh,lkmvh->vnklm',orthoradint1[iat,ispe],harmconj[iat,ispe])
        

                power = np.zeros((nat[i],2*lam+1,nspecies,nspecies,nmax,nmax,lmax+1,lmax+1),complex)
                p2 = np.zeros((nat[i],2*lam+1,nspecies,nspecies,nmax,nmax,llmax),complex)
                for iat in range(ncen[i]):
                    for ia in range(nspecies):
                        for ib in range(nspecies):
                            power[iat,:,ia,ib] = np.einsum('nlv,xmlkv,xlkv->xnmlk',omega2[iat,ia],omegaconj[iat,ib],w3j)

                for l in range(llmax):
                    p2[:,:,:,:,:,:,l] = power[:,:,:,:,:,:,lvalues[l][0],lvalues[l][1]]
                p2 = p2.reshape(nat[i],2*lam+1,featsize)
                PS[i,:nat[i]] = p2[:,:,do_list]


            else:

                featsize2 = len(do_list)
                power = np.zeros((nat[i],2*lam+1,featsize2),dtype=complex)
                harmconj = np.zeros((nat[i],nspecies,lmax+1,lmax+1,2*lmax+1,2*lam+1,nnmax),dtype=complex)
                for cc in range(len(keep_components)):
                    [ia,ib,nn,mm,l1,l2] = keep_components[cc]
                    for iat in range(ncen[i]):
                        for lm in range(2*lam+1):
                            for im in range(2*l1+1):
                                if abs(im-l1-lm+lam) <= l2:
                                    harmconj[:,:,l2,l1,im,lm,:] = np.conj(harmonic1[:,:,l2,l2+im-l1-lm+lam,:])
                            power[iat,lm,cc] = np.einsum('a,b,ab,a',omega2[iat,ia,nn,l1,:],orthoradint1[iat,ib,l2,mm,:],harmconj[iat,ib,l2,l1,:,lm,:],w3j[lm,l1,l2,:])
                PS[i,:nat[i]] = power

    # Multiply by A_matrix.
    if (sparse_options[0] != ''):
        PS = np.dot(PS,sparse_options[2])

    return [PS,featsize]

##################################################################################################################

def sparsify(PS,featsize,ncut):
    """sparsify power spectrum with PCA"""

    eigenvalues, unitary = np.linalg.eigh(np.dot(PS.T,np.conj(PS)))
    psparse = np.dot(PS,unitary[:,featsize-ncut:featsize])

    return psparse 

##################################################################################################################

def FPS_sparsify(PS,featsize,ncut,initial):
    """Sparsify power spectrum with FPS"""

    # Get FPS vector.
    if (ncut>featsize):
        ncut = featsize
    vec_fps = do_fps(PS.T,ncut,initial)
    # Get A matrix.
    C_matr = PS[:,vec_fps]
    UR = np.dot(np.linalg.pinv(C_matr),PS)
    ururt = np.dot(UR,np.conj(UR.T))
    [eigenvals,eigenvecs] = np.linalg.eigh(ururt)
    print("Lowest eigenvalue = %f"%eigenvals[0])
    eigenvals = np.array([np.sqrt(max(eigenvals[i],0)) for i in range(len(eigenvals))])
    diagmatr = np.diag(eigenvals)
    A_matrix = np.dot(np.dot(eigenvecs,diagmatr),eigenvecs.T)

    # Sparsify the matrix by taking the requisite columns.
    psparse = np.array([PS.T[i] for i in vec_fps]).T
    psparse = np.dot(psparse,A_matrix)

    # Return the sparsification vector (which we will need for later sparsification) and the A matrix (which we will need for recombination).
    sparse_details = [vec_fps,A_matrix]

    return [psparse,sparse_details]

##################################################################################################################

def do_fps(x, d=0,initial=-1):
    # Code from Giulio Imbalzano

    if d == 0 : d = len(x)
    n = len(x)
    iy = np.zeros(d,int)
    if (initial == -1):
        iy[0] = np.random.randint(0,n)
    else:
        iy[0] = initial
    # Faster evaluation of Euclidean distance
    # Here we fill the n2 array in this way because it halves the memory cost of this routine
    n2 = np.array([np.sum(x[i] * np.conj([x[i]])) for i in range(len(x))])
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
    for i in range(1,d):
        print("Doing ",i," of ",d," dist = ",max(dl))
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

##################################################################################################################
