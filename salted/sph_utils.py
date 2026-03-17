import sys
import math
import time
from typing import Tuple

import numpy as np
from scipy import special
from ase.data import atomic_numbers

from featomic import SphericalExpansion
from featomic import LodeSphericalExpansion
from metatensor import Labels

from numba import njit, prange

def cartesian_to_spherical_transformation(l):
        """Compute Cartesian to spherical transformation matrices sorting the spherical components as {-l,..,0,..,+l} 
        while sorting the Cartesian components as shown in the corresponding arrays."""

        if l==0:
            # 1 Cartesian triplet
            cart_idx = [[0,0,0]]
        elif l==1:
            # 3 Cartesian triplets
            cart_idx = [[1,0,0],[0,1,0],[0,0,1]]
        elif l==2:
            # 6 Cartesian triplets
            cart_idx = [[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]
        elif l==3:
            # 10 Cartesian triplets
            cart_idx = [[3,0,0],[2,1,0],[2,0,1],[1,2,0],[1,1,1],[1,0,2],[0,3,0],[0,2,1],[0,1,2],[0,0,3]]
        elif l==4:
            # 15 Cartesian triplets
            cart_idx = [[4,0,0],[3,1,0],[3,0,1],[2,2,0],[2,1,1],[2,0,2],[1,3,0],[1,2,1],[1,1,2],[1,0,3],[0,4,0],[0,3,1],[0,2,2],[0,1,3],[0,0,4]]
        elif l==5:
            # 21 Cartesian triplets
            cart_idx = [[0,0,5],[2,0,3],[0,2,3],[4,0,1],[0,4,1],[2,2,1],[1,0,4],[0,1,4],[3,0,2],[0,3,2],[1,2,2],[2,1,2],[5,0,0],[0,5,0],[1,4,0],[4,1,0],[3,2,0],[2,3,0],[1,1,3],[3,1,1],[1,3,1]]
        elif l==6:
            # 28 Cartesian triplets
            cart_idx = [[6,0,0],[0,6,0],[0,0,6],[5,0,1],[5,1,0],[0,5,1],[1,5,0],[0,1,5],[1,0,5],[4,0,2],[4,2,0],[0,4,2],[2,4,0],[0,2,4],[2,0,4],[4,1,1],[1,4,1],[1,1,4],[3,1,2],[1,3,2],[1,2,3],[3,2,1],[2,3,1],[2,1,3],[3,3,0],[3,0,3],[0,3,3],[2,2,2]]
        else:
            print("ERROR: Cartesian to spherical transformation not available for l=",l)

        mat = np.zeros((2*l+1,int((l+2)*(l+1)/2)),complex)
        # this implementation follows Eq.15 of SCHLEGEL and FRISH, Inter. J. Quant. Chem., Vol. 54, 83-87 (1995)
        for m in range(l+1):
            itriplet = 0
            for triplet in cart_idx:
                lx = triplet[0]
                ly = triplet[1]
                lz = triplet[2]
                sfact = np.sqrt(math.factorial(l)*math.factorial(2*lx)*math.factorial(2*ly)*math.factorial(2*lz)*math.factorial(l-m)/(math.factorial(lx)*math.factorial(ly)*math.factorial(lz)*math.factorial(2*l)*math.factorial(l+m))) / (math.factorial(l)*2**l)
                j = (lx+ly-m)/2
                if j.is_integer()==True:
                    j = int(j)
                    if j>=0:
                       for ii in range(math.floor((l-m)/2)+1):
                           if j<=ii:
                               afact = special.binom(l,ii)*special.binom(ii,j)*math.factorial(2*l-2*ii)/math.factorial(l-m-2*ii)*(-1.0)**ii
                               for k in range(j+1):
                                   kk = lx-2*k
                                   if m>=kk and kk>=0:
                                      jj = (m-kk)/2
                                      bfact = special.binom(j,k)*special.binom(m,kk)*(-1.0)**(jj)
                                      mat[l+m,itriplet] += afact*bfact
                mat[l+m,itriplet] *= sfact
                mat[l-m,itriplet] = np.conj(mat[l+m,itriplet])
                if m%2!=0:
                    mat[l+m,itriplet] *= -1.0 # TODO convention to be understood
                itriplet += 1

        return[np.asarray(mat), cart_idx]

def complex_to_real_transformation(sizes):
    """Transformation matrix from complex to real spherical harmonics"""

    matrices = []
    for i in range(len(sizes)):
        lval = int((sizes[i]-1)/2)
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range(lval):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[lval][lval] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

def get_angular_indexes_symmetric(lam,nang1,nang2) -> Tuple[int, np.ndarray]:
    """Select relevant angular indexes for equivariant descriptor calculation"""

    llmax = 0
    lvalues = {}

    for l1 in range(nang1+1):
        for l2 in range(nang2+1):
            # keep only even combination to enforce inversion symmetry
            if (lam+l1+l2)%2==0 :
                # enforce triangular inequality 
                if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                    lvalues[llmax] = [l1,l2]
                    llmax+=1

    # Fill dense array from dictionary
    llvec = np.zeros((llmax,2),int)
    for il in range(llmax):
        llvec[il,0] = lvalues[il][0]
        llvec[il,1] = lvalues[il][1]

    return llmax, llvec

def get_angular_indexes_antisymmetric(lam,nang1,nang2) -> Tuple[int, np.ndarray]:
    """Select relevant angular indexes for equivariant descriptor calculation, antisymmetric with respect to inversion operations"""

    llmax = 0
    lvalues = {}

    for l1 in range(nang1+1):
        for l2 in range(nang2+1):
            # keep only even combination to enforce inversion symmetry
            if (lam+l1+l2)%2!=0 :
                # enforce triangular inequality 
                if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                    lvalues[llmax] = [l1,l2]
                    llmax+=1

    # Fill dense array from dictionary
    llvec = np.zeros((llmax,2),int)
    for il in range(llmax):
        llvec[il,0] = lvalues[il][0]
        llvec[il,1] = lvalues[il][1]

    return llmax, llvec

def get_representation_coeffs(structure,rep,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe,species,nang,nrad,natoms):
    """Compute spherical harmonics expansion coefficients of the given structural representation."""

    if rep=="rho":

        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep=="V":

        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:

        if rank == 0: print("Error: requested representation", rep, "not provided")

    nspe = len(neighspe)
    keys_array = np.zeros(((nang+1)*len(species)*nspe,4),int)
    i = 0
    for l in range(nang+1):
        for specen in species:
            for speneigh in neighspe:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx = calculator.compute(structure, selected_keys=keys_selection)
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega = np.zeros((nang+1,natoms,2*nang+1,nspe*nrad),complex)
    for l in range(nang+1):
        c2r = complex_to_real_transformation([2*l+1])[0]
        omega[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)

    return omega

def get_representation_coeffs_atomrange(structure,rep,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe,species,nang,nrad,atoms_range):
    """Compute spherical harmonics expansion coefficients of the given structural representation."""

    if rep=="rho":

        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep=="V":

        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:

        if rank == 0: print("Error: requested representation", rep, "not provided")

    nspe = len(neighspe)
    keys_array = np.zeros(((nang+1)*len(species)*nspe,4),int)
    i = 0
    for l in range(nang+1):
        for specen in species:
            for speneigh in neighspe:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    natoms_range = len(atoms_range)
    samples_selection = Labels(
        names=["system","atom"],
        values=np.column_stack([
                np.zeros(natoms_range, dtype=int),  # system index
                atoms_range                         # atom indices
            ])
    )

    spx = calculator.compute(structure, selected_keys=keys_selection, selected_samples=samples_selection)
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega = np.zeros((nang+1,natoms_range,2*nang+1,nspe*nrad),complex)
    for l in range(nang+1):
        c2r = complex_to_real_transformation([2*l+1])[0]
        omega[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)

    return omega

def get_representation_gradient_coeffs(structure,rep,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe,species,nang,nrad,natoms):
    """Compute spherical harmonics expansion coefficients of the given structural representation."""

    if rep=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0: print("Error: requested representation", rep, "not provided")

    nspe = len(neighspe)
    keys_array = np.zeros(((nang+1)*len(species)*nspe,4),int)
    i = 0
    for l in range(nang+1):
        for specen in species:
            for speneigh in neighspe:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx = calculator.compute(structure, selected_keys=keys_selection, gradients=["positions"])
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega = np.zeros((nang+1,natoms,2*nang+1,nspe*nrad),complex)
    for l in range(nang+1):
        c2r = complex_to_real_transformation([2*l+1])[0]
        omega[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)
        #omega[l,:,:2*l+1,:] = np.transpose(np.dot(np.conj(c2r.T),np.transpose(spx.block(o3_lambda=l).values, (1,0,2)).reshape((np.conj(c2r.T).shape[1], natoms_range*nspe*nrad))).reshape(2*l+1,natoms_range,nspe*nrad),(1,0,2))

    # Get 1st set of coefficients as a complex numpy array
    domega = np.zeros((nang+1,natoms*natoms,3,2*nang+1,nspe*nrad),complex)
    for l in range(nang+1):
        grad = spx.block(o3_lambda=l).gradient("positions")
        c2r = complex_to_real_transformation([2*l+1])[0]
        if grad.values.shape[0] != natoms*natoms:

            # Extract indices as arrays
            iat = grad.samples["sample"]      # shape (nidx,)
            i_grad = grad.samples["atom"]     # shape (nidx,)
            target_idx = iat * natoms + i_grad

            # grad.values shape: (nidx, k, r, d)
            # Apply transformation to ALL indices at once
            domega[l, target_idx, :, :2*l+1, :] = np.transpose(np.dot(np.conj(c2r.T),np.transpose(grad.values,(2,0,1,3)).reshape((np.conj(c2r.T).shape[1],len(target_idx)*3*nspe*nrad))).reshape((2*l+1,len(target_idx),3,nspe*nrad)),(1,2,0,3))

        else:
            domega[l, :, :, :2*l+1, :] = np.transpose(np.dot(np.conj(c2r.T),np.transpose(grad.values,(2,0,1,3)).reshape((np.conj(c2r.T).shape[1],natoms*natoms*3*nspe*nrad))).reshape((2*l+1,natoms*natoms,3,nspe*nrad)),(1,2,0,3))

    return omega, domega

def get_representation_gradient_coeffs_atomrange(structure,rep,HYPER_PARAMETERS_DENSITY,HYPER_PARAMETERS_POTENTIAL,rank,neighspe,species,nang,nrad,natoms,atoms_range):
    """Compute spherical harmonics expansion coefficients of the given structural representation."""

    if rep=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        if rank == 0: print("Error: requested representation", rep, "not provided")

    nspe = len(neighspe)
    keys_array = np.zeros(((nang+1)*len(species)*nspe,4),int)
    i = 0
    for l in range(nang+1):
        for specen in species:
            for speneigh in neighspe:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    natoms_range = len(atoms_range)

    samples_selection = Labels(
        names=["system","atom"],
        values=np.column_stack([
                np.zeros(natoms_range, dtype=int),  # system index
                atoms_range                         # atom indices
            ])
    )

    spx = calculator.compute(structure, selected_keys=keys_selection, selected_samples=samples_selection, gradients=["positions"])
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega = np.zeros((nang+1,natoms_range,2*nang+1,nspe*nrad),complex)
    domega = np.zeros((nang+1,natoms_range*natoms,3,2*nang+1,nspe*nrad),complex)

    for l in range(nang+1):

        c2r = complex_to_real_transformation([2*l+1])[0]
        
        omega[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)

        grad = spx.block(o3_lambda=l).gradient("positions")
        c2r = complex_to_real_transformation([2*l+1])[0]

        if grad.values.shape[0] != natoms_range*natoms:

            # Extract indices as arrays
            iat = grad.samples["sample"]      # shape (nidx,)
            i_grad = grad.samples["atom"]    # shape (nidx,)

            target_idx = iat * natoms + i_grad

            domega[l, target_idx, :, :2*l+1, :] = np.transpose(np.dot(np.conj(c2r.T),np.transpose(grad.values,(2,0,1,3)).reshape((np.conj(c2r.T).shape[1],len(target_idx)*3*nspe*nrad))).reshape((2*l+1,len(target_idx),3,nspe*nrad)),(1,2,0,3))

        else:

            domega[l, :, :, :2*l+1, :] = np.transpose(np.dot(np.conj(c2r.T),np.transpose(grad.values,(2,0,1,3)).reshape((np.conj(c2r.T).shape[1],natoms_range*natoms*3*nspe*nrad))).reshape((2*l+1,natoms_range*natoms,3,nspe*nrad)),(1,2,0,3))
    
    return omega, domega

@njit(parallel=True, fastmath = True)
def grad_equicombsparse_numba(natoms,natoms_range,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,w3j,llmax,llvec,lam,c2r,featsize,nfps,vfps):
    p = np.zeros((natoms_range, 2*lam+1, nfps), dtype=np.float64)
    grad_p = np.zeros((natoms, 3, natoms_range, 2*lam+1, nfps), dtype=np.float64)
    dv2c = np.conj(dv2)
    v2c  = np.conj(v2)

    for iat in prange(natoms_range):
        inner = 0.0
        dot1 = np.zeros(natoms, dtype=np.float64)
        dot2 = np.zeros(natoms, dtype=np.float64)
        dot3 = np.zeros(natoms, dtype=np.float64)
        ptemp = np.zeros((featsize, 2*lam+1), dtype=np.float64)
        grad_ptemp = np.zeros((featsize, 2*lam+1, natoms, 3), dtype=np.float64)
        ifeat = 0
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l1 = llvec[il,0]
                    l2 = llvec[il,1]
                    pcmplx = np.zeros(2*lam+1, dtype=np.complex128)
                    grad_pcmplx = np.zeros((2*lam+1, natoms, 3), dtype=np.complex128)
                    for imu in range(2*lam+1):
                        mu = imu-lam
                        for im1 in range(2*l1+1):
                            m1 = im1-l1
                            m2 = m1-mu
                            if (abs(m2)<=l2):
                               im2 = m2+l2
                               v2cv  = v2c[iat,n2,l2,im2]
                               v1v = v1[iat,n1,l1,im1]
                               pcmplx[imu] = pcmplx[imu] + w3j[iwig] * v1v * v2cv
                               for i_grad in range(natoms):
                                   grad_pcmplx[imu,i_grad,0] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,0] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,0])
                                   grad_pcmplx[imu,i_grad,1] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,1] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,1])
                                   grad_pcmplx[imu,i_grad,2] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,2] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,2])
                               iwig = iwig + 1
                    preal = np.zeros(2*lam+1, dtype=np.float64)
                    grad_preal = np.zeros((2*lam+1, natoms, 3), dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                             preal[imu] = preal[imu] + np.real(c2r[imu,im1] * pcmplx[im1])
                             for i_grad in range(natoms):
                                  grad_preal[imu,i_grad,0] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,0])
                                  grad_preal[imu,i_grad,1] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,1])
                                  grad_preal[imu,i_grad,2] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,2])
                        inner = inner + preal[imu]**2
                        ptemp[ifeat,imu] = preal[imu]
                        for i_grad in range(natoms):
                            dot1[i_grad] += preal[imu]*grad_preal[imu,i_grad,0]
                            dot2[i_grad] += preal[imu]*grad_preal[imu,i_grad,1]
                            dot3[i_grad] += preal[imu]*grad_preal[imu,i_grad,2]
                            grad_ptemp[ifeat,imu,i_grad,0] = grad_preal[imu,i_grad,0]
                            grad_ptemp[ifeat,imu,i_grad,1] = grad_preal[imu,i_grad,1]
                            grad_ptemp[ifeat,imu,i_grad,2] = grad_preal[imu,i_grad,2]
                    ifeat = ifeat + 1
        normfact = np.sqrt(inner)
        normfact3 = normfact**3
        for n in range(nfps):
            ifps = vfps[n]
            for imu in range(2*lam+1):
                p[iat,imu,n] = ptemp[ifps,imu] / normfact
                for i_grad in range(natoms):
                    grad_p[i_grad,0,iat,imu,n] = (grad_ptemp[ifps,imu,i_grad,0] / normfact) - ptemp[ifps,imu] * dot1[i_grad] / (normfact3)
                    grad_p[i_grad,1,iat,imu,n] = (grad_ptemp[ifps,imu,i_grad,1] / normfact) - ptemp[ifps,imu] * dot2[i_grad] / (normfact3)
                    grad_p[i_grad,2,iat,imu,n] = (grad_ptemp[ifps,imu,i_grad,2] / normfact) - ptemp[ifps,imu] * dot3[i_grad] / (normfact3)
    return p, grad_p

@njit(parallel=True, fastmath = True)
def grad_equicomb_numba(natoms,natoms_range,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,w3j,llmax,llvec,lam,c2r,featsize):
    p = np.zeros((natoms_range, 2*lam+1, featsize), dtype=np.float64)
    grad_p = np.zeros((natoms, 3, natoms_range, 2*lam+1, featsize), dtype=np.float64)
    dv2c = np.conj(dv2)
    v2c  = np.conj(v2)

    for iat in prange(natoms_range):
        inner = 0.0
        dot1 = np.zeros(natoms, dtype=np.float64)
        dot2 = np.zeros(natoms, dtype=np.float64)
        dot3 = np.zeros(natoms, dtype=np.float64)
        ptemp = np.zeros((featsize, 2*lam+1), dtype=np.float64)
        grad_ptemp = np.zeros((featsize, 2*lam+1, natoms, 3), dtype=np.float64)
        ifeat = 0
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l1 = llvec[il,0]
                    l2 = llvec[il,1]
                    pcmplx = np.zeros(2*lam+1, dtype=np.complex128)
                    grad_pcmplx = np.zeros((2*lam+1, natoms, 3), dtype=np.complex128)
                    for imu in range(2*lam+1):
                        mu = imu-lam
                        for im1 in range(2*l1+1):
                            m1 = im1-l1
                            m2 = m1-mu
                            if (abs(m2)<=l2):
                               im2 = m2+l2
                               v2cv  = v2c[iat,n2,l2,im2]
                               v1v = v1[iat,n1,l1,im1]
                               pcmplx[imu] = pcmplx[imu] + w3j[iwig] * v1v * v2cv
                               for i_grad in range(natoms):
                                   grad_pcmplx[imu,i_grad,0] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,0] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,0])
                                   grad_pcmplx[imu,i_grad,1] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,1] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,1])
                                   grad_pcmplx[imu,i_grad,2] += w3j[iwig] * (dv1[iat,n1,l1,im1,i_grad,2] * v2cv + v1v * dv2c[iat,n2,l2,im2,i_grad,2])
                               iwig = iwig + 1
                    preal = np.zeros(2*lam+1, dtype=np.float64)
                    grad_preal = np.zeros((2*lam+1, natoms, 3), dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                             preal[imu] = preal[imu] + np.real(c2r[imu,im1] * pcmplx[im1])
                             for i_grad in range(natoms):
                                  grad_preal[imu,i_grad,0] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,0])
                                  grad_preal[imu,i_grad,1] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,1])
                                  grad_preal[imu,i_grad,2] += np.real(c2r[imu,im1] * grad_pcmplx[im1,i_grad,2])
                        inner = inner + preal[imu]**2
                        ptemp[ifeat,imu] = preal[imu]
                        for i_grad in range(natoms):
                            dot1[i_grad] += preal[imu]*grad_preal[imu,i_grad,0]
                            dot2[i_grad] += preal[imu]*grad_preal[imu,i_grad,1]
                            dot3[i_grad] += preal[imu]*grad_preal[imu,i_grad,2]
                            grad_ptemp[ifeat,imu,i_grad,0] = grad_preal[imu,i_grad,0]
                            grad_ptemp[ifeat,imu,i_grad,1] = grad_preal[imu,i_grad,1]
                            grad_ptemp[ifeat,imu,i_grad,2] = grad_preal[imu,i_grad,2]
                    ifeat = ifeat + 1
        normfact = np.sqrt(inner)
        normfact3 = normfact**3
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu] / normfact
                for i_grad in range(natoms):
                    grad_p[i_grad,0,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,0] / normfact) - ptemp[ifeat,imu] * dot1[i_grad] / (normfact3)
                    grad_p[i_grad,1,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,1] / normfact) - ptemp[ifeat,imu] * dot2[i_grad] / (normfact3)
                    grad_p[i_grad,2,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,2] / normfact) - ptemp[ifeat,imu] * dot3[i_grad] / (normfact3)
    return p, grad_p

@njit(parallel=True, fastmath = True)
def equicombsparse_numba(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize,nfps,vfps):
    p = np.zeros((natoms, 2*lam+1, nfps), dtype=np.float64)
    v2c  = np.conj(v2)

    for iat in prange(natoms):
        inner = 0.0
        ptemp = np.zeros((featsize, 2*lam+1), dtype=np.float64)
        ifeat = 0
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l1 = llvec[il,0]
                    l2 = llvec[il,1]
                    pcmplx = np.zeros(2*lam+1, dtype=np.complex128)
                    for imu in range(2*lam+1):
                        mu = imu-lam
                        for im1 in range(2*l1+1):
                            m1 = im1-l1
                            m2 = m1-mu
                            if (abs(m2)<=l2):
                               im2 = m2+l2
                               v2cv  = v2c[iat,n2,l2,im2]
                               v1v = v1[iat,n1,l1,im1]
                               pcmplx[imu] = pcmplx[imu] + w3j[iwig] * v1v * v2cv
                               iwig = iwig + 1
                    preal = np.zeros(2*lam+1, dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                             preal[imu] = preal[imu] + np.real(c2r[imu,im1] * pcmplx[im1])
                        inner = inner + preal[imu]**2
                        ptemp[ifeat,imu] = preal[imu]
                    ifeat = ifeat + 1
        normfact = np.sqrt(inner)
        for n in range(nfps):
            ifps = vfps[n]
            for imu in range(2*lam+1):
                p[iat,imu,n] = ptemp[ifps,imu] / normfact
    return p

@njit(parallel=True, fastmath = True)
def equicomb_numba(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize):
    p = np.zeros((natoms, 2*lam+1, featsize), dtype=np.float64)
    v2c  = np.conj(v2)

    for iat in prange(natoms):
        inner = 0.0
        ptemp = np.zeros((featsize, 2*lam+1), dtype=np.float64)
        ifeat = 0
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l1 = llvec[il,0]
                    l2 = llvec[il,1]
                    pcmplx = np.zeros(2*lam+1, dtype=np.complex128)
                    for imu in range(2*lam+1):
                        mu = imu-lam
                        for im1 in range(2*l1+1):
                            m1 = im1-l1
                            m2 = m1-mu
                            if (abs(m2)<=l2):
                               im2 = m2+l2
                               v2cv  = v2c[iat,n2,l2,im2]
                               v1v = v1[iat,n1,l1,im1]
                               pcmplx[imu] = pcmplx[imu] + w3j[iwig] * v1v * v2cv
                               iwig = iwig + 1
                    preal = np.zeros(2*lam+1, dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                             preal[imu] = preal[imu] + np.real(c2r[imu,im1] * pcmplx[im1])
                        inner = inner + preal[imu]**2
                        ptemp[ifeat,imu] = preal[imu]
                    ifeat = ifeat + 1
        normfact = np.sqrt(inner)
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu] / normfact
    return p
