import sys
import math
import time
import warnings
import numpy as np
from scipy import special
from ase import Atoms
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

def get_angular_indexes_symmetric(lam,nang1,nang2) -> tuple[int, np.ndarray]:
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

def get_angular_indexes_antisymmetric(lam,nang1,nang2) -> tuple[int, np.ndarray]:
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

def reps_equivalent(
    rep1: str, neighspe1: list[str], hyper_params1: dict,
    rep2: str, neighspe2: list[str], hyper_params2: dict,
) -> bool:
    """Return True if two rep configs are identical (same type, neighbor species, and hyperparameters).

    When True, the caller can reuse omega1 for omega2 instead of recomputing.
    """
    return rep1 == rep2 and list(neighspe1) == list(neighspe2) and hyper_params1 == hyper_params2


def get_representation_coeffs(
    structure: Atoms, rep: str, hyper_params: dict,
    rank: int, neighspe: list[str], species: list[str],
    nang: int, nrad: int, natoms: int,
) -> np.ndarray:
    """Compute spherical harmonics expansion coefficients of the given structural representation.

    Args:
        structure (Atoms): ASE Atoms object.
        rep (str): representation type, "rho" (SOAP) or "V" (LODE).
        hyper_params (dict): featomic hyperparameter dict for this representation,
                             as returned by sys_utils.build_featomic_hyper_params().
        rank (int): MPI rank (used for error messages).
        neighspe (list[str]): neighbor species list.
        species (list[str]): center species list.
        nang (int): max angular channel.
        nrad (int): number of radial basis functions.
        natoms (int): number of atoms in the structure.
    """

    if rep=="rho":

        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**hyper_params)

    elif rep=="V":

        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**hyper_params)

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

def get_representation_coeffs_atomrange(
    structure: Atoms, rep: str, hyper_params: dict,
    rank: int, neighspe: list[str], species: list[str],
    nang: int, nrad: int, atoms_range: np.ndarray,
) -> np.ndarray:
    """Compute spherical harmonics expansion coefficients of the given structural representation.

    Args:
        structure (Atoms): ASE Atoms object.
        rep (str): representation type, "rho" (SOAP) or "V" (LODE).
        hyper_params (dict): featomic hyperparameter dict for this representation,
                             as returned by sys_utils.build_featomic_hyper_params().
        rank (int): MPI rank (used for error messages).
        neighspe (list[str]): neighbor species list.
        species (list[str]): center species list.
        nang (int): max angular channel.
        nrad (int): number of radial basis functions.
        atoms_range (np.ndarray): indices of atoms to compute.
    """

    if rep=="rho":

        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**hyper_params)

    elif rep=="V":

        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**hyper_params)

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

def get_representation_gradient_coeffs(
    structure: Atoms, rep: str, hyper_params: dict,
    rank: int, neighspe: list[str], species: list[str],
    nang: int, nrad: int, natoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spherical harmonics expansion coefficients and their position gradients.

    Args:
        structure (Atoms): ASE Atoms object.
        rep (str): representation type, "rho" (SOAP) or "V" (LODE).
        hyper_params (dict): featomic hyperparameter dict for this representation,
                             as returned by sys_utils.build_featomic_hyper_params().
        rank (int): MPI rank (used for error messages).
        neighspe (list[str]): neighbor species list.
        species (list[str]): center species list.
        nang (int): max angular channel.
        nrad (int): number of radial basis functions.
        natoms (int): number of atoms in the structure.
    """

    if rep=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**hyper_params)

    elif rep=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**hyper_params)

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

def get_representation_gradient_coeffs_atomrange(
    structure: Atoms, rep: str, hyper_params: dict,
    rank: int, neighspe: list[str], species: list[str],
    nang: int, nrad: int, natoms: int, atoms_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute spherical harmonics expansion coefficients and position gradients for a subset of atoms.

    Args:
        structure (Atoms): ASE Atoms object.
        rep (str): representation type, "rho" (SOAP) or "V" (LODE).
        hyper_params (dict): featomic hyperparameter dict for this representation,
                             as returned by sys_utils.build_featomic_hyper_params().
        rank (int): MPI rank (used for error messages).
        neighspe (list[str]): neighbor species list.
        species (list[str]): center species list.
        nang (int): max angular channel.
        nrad (int): number of radial basis functions.
        natoms (int): total number of atoms in the structure.
        atoms_range (np.ndarray): indices of atoms to compute.
    """

    if rep=="rho":
        # get SPH expansion for atomic density
        calculator = SphericalExpansion(**hyper_params)

    elif rep=="V":
        # get SPH expansion for atomic potential
        calculator = LodeSphericalExpansion(**hyper_params)

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
def grad_equicombsparse_numba(natoms,natoms_range,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,w3j,llmax,llvec,lam,c2r,featsize,nfps,vfps,norm0,dot0):
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
        if norm0.size > 0:
            # rel_power: lam > 0 is divided by |q0| and differentiated with q0 . dq0
            if lam > 0:
                normfact = norm0[iat]
                for i_grad in range(natoms):
                    dot1[i_grad] = dot0[iat,i_grad,0]
                    dot2[i_grad] = dot0[iat,i_grad,1]
                    dot3[i_grad] = dot0[iat,i_grad,2]
            if normfact == 0.0:
                continue
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
def grad_equicomb_numba(natoms,natoms_range,nang1,nang2,nrad1,nrad2,v1,v2,dv1,dv2,w3j,llmax,llvec,lam,c2r,featsize,norm0,dot0):
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
        if norm0.size > 0:
            # rel_power: lam > 0 is divided by |q0| and differentiated with q0 . dq0
            if lam > 0:
                normfact = norm0[iat]
                for i_grad in range(natoms):
                    dot1[i_grad] = dot0[iat,i_grad,0]
                    dot2[i_grad] = dot0[iat,i_grad,1]
                    dot3[i_grad] = dot0[iat,i_grad,2]
            if normfact == 0.0:
                continue
        normfact3 = normfact**3
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu] / normfact
                for i_grad in range(natoms):
                    grad_p[i_grad,0,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,0] / normfact) - ptemp[ifeat,imu] * dot1[i_grad] / (normfact3)
                    grad_p[i_grad,1,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,1] / normfact) - ptemp[ifeat,imu] * dot2[i_grad] / (normfact3)
                    grad_p[i_grad,2,iat,imu,ifeat] = (grad_ptemp[ifeat,imu,i_grad,2] / normfact) - ptemp[ifeat,imu] * dot3[i_grad] / (normfact3)
    return p, grad_p

def _njit_cached(**options):
    """njit with cache=True; numba raises at decoration when it has no writable
    cache location, and the function is then compiled in every process."""
    def decorate(func):
        try:
            return njit(cache=True, **options)(func)
        except RuntimeError as err:
            if "no locator available" not in str(err):
                raise
            warnings.warn("numba cannot write its cache here; SALTED compiles its sparse "
                          "equicomb kernel in every run. Set NUMBA_CACHE_DIR to a writable folder.")
            return njit(**options)(func)
    return decorate


@_njit_cached(fastmath = True)
def _density_matrix(v, iat, nang, nrad):
    """Lower triangle (m >= m') of A[l, m, m'] = sum_n v[n, l, m] conj(v[n, l, m'])
    of one atom.  Only m >= m' is filled."""
    nm = 2*nang+1
    A = np.empty((nang+1, nm, nm), dtype=np.complex128)
    X = np.empty((nm, nrad), dtype=np.complex128)
    for l in range(nang+1):
        # X[m, n] = v[n, l, m]: the sums over n run over contiguous memory
        for n in range(nrad):
            for i in range(2*l+1):
                X[i,n] = v[iat,n,l,i]
        # A[-m', -m] = (-1)^(m+m') A[m, m'] for a real density
        for i in range(l+1):
            for j in range(i, 2*l+1-i):
                s = 0j
                for n in range(nrad):
                    s += X[j,n] * np.conj(X[i,n])
                A[l,j,i] = s
                A[l,2*l-i,2*l-j] = (1.0 - 2.0*((i+j) % 2)) * s
        # the diagonal of m > 0 summed directly, so every coefficient enters the trace
        for i in range(l+1, 2*l+1):
            d = 0.0
            for n in range(nrad):
                d += X[i,n].real * X[i,n].real + X[i,n].imag * X[i,n].imag
            A[l,i,i] = d
    return A


@_njit_cached(fastmath = False)
def _finite(x):
    """x is neither NaN nor inf"""
    return np.isfinite(x)


@_njit_cached()
def _same_atom(v1, v2, iat):
    """v1[iat] == v2[iat], compared on m <= 0 (m > 0 follows for a real density)"""
    for n in range(v1.shape[1]):
        for l in range(v1.shape[2]):
            for m in range(l+1):
                if v1[iat,n,l,m] != v2[iat,n,l,m]:
                    return False
    return True


@_njit_cached(fastmath = True)
def _norm2(W, llvec, lam, A1, A2):
    """sum over all features of |pc|^2 from the density matrices:
    sum_il sum_mu sum_{m1, m1'} W[il, mu, m1] W[il, mu, m1']
    Re(A1[l1, m1, m1'] conj(A2[l2, m1 - mu, m1' - mu])),
    and its upper bound sum_il tr(A1[l1]) tr(A2[l2]) / (2 lam + 1)."""
    inner = 0.0
    bound = 0.0
    for il in range(llvec.shape[0]):
        l1, l2 = llvec[il,0], llvec[il,1]
        tr1 = 0.0
        for i in range(2*l1+1):
            tr1 += A1[l1,i,i].real
        tr2 = 0.0
        for j in range(2*l2+1):
            tr2 += A2[l2,j,j].real
        bound += tr1 * tr2
        for mu in range(lam+1):
            # im1 = i0+t pairs with im2 = j0+t (m2 = m1-mu), t = 0..cnt-1
            i0 = max(0, l1+mu-l2)
            j0 = max(0, l2-l1-mu)
            cnt = min(2*l1-i0, 2*l2-j0)+1
            # s is half the sum over (m1, m1'): diagonal halved, each pair m1 > m1' once
            s = 0.0
            for t in range(cnt):
                x = 0.5 * W[il,mu,i0+t] * A1[l1,i0+t,i0+t].real * A2[l2,j0+t,j0+t].real
                for u in range(t):
                    x += W[il,mu,i0+u] * (A1[l1,i0+t,i0+u] * np.conj(A2[l2,j0+t,j0+u])).real
                s += W[il,mu,i0+t] * x
            # mu and -mu give the same term
            inner += (2.0 if mu == 0 else 4.0) * s
    return inner, bound / (2*lam+1)


@_njit_cached(fastmath = False)
def _norm2_direct(W, cpm, llvec, lam, v1, v2, iat, nrad1, nrad2):
    """sum of the squares of all features of one atom, each feature computed
    as in the feature loop of equicombsparse_numba, in strict floating point so
    that a compiled and a cache-loaded kernel give the same sum"""
    inner = 0.0
    for il in range(llvec.shape[0]):
        l1, l2 = llvec[il,0], llvec[il,1]
        i0 = max(0, l1-l2)
        j0 = max(0, l2-l1)
        cnt = min(l1, l2)
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                x = 0.0
                for t in range(cnt):
                    x += W[il,0,i0+t] * (v1[iat,n1,l1,i0+t] * np.conj(v2[iat,n2,l2,j0+t])).real
                x = 2*x + W[il,0,l1] * v1[iat,n1,l1,l1].real * v2[iat,n2,l2,l2].real
                q = cpm[lam].real * x
                inner += q * q
        for mu in range(1, lam+1):
            i0 = max(0, l1+mu-l2)
            j0 = max(0, l2-l1-mu)
            cnt = min(2*l1-i0, 2*l2-j0)+1
            for n1 in range(nrad1):
                for n2 in range(nrad2):
                    s = 0j
                    for t in range(cnt):
                        s += W[il,mu,i0+t] * v1[iat,n1,l1,i0+t] * np.conj(v2[iat,n2,l2,j0+t])
                    qp = (cpm[lam+mu] * s).real
                    qm = (cpm[lam-mu] * s).real
                    inner += qp * qp + qm * qm
    return inner


@_njit_cached(fastmath = True)
def _wigner_tables(w3j, llmax, llvec, lam, c2r, nang1):
    """W[il, mu, im1]: w3j of mu >= 0 in the order the per-feature loop reads it,
    and cpm with q[imu] = Re(cpm[imu] * pc[|mu|])"""
    nmu = 2*lam+1
    W = np.zeros((llmax, lam+1, 2*nang1+1))
    iwig = 0
    for il in range(llmax):
        l1, l2 = llvec[il,0], llvec[il,1]
        for imu in range(nmu):
            for im1 in range(2*l1+1):
                if abs(im1-l1-(imu-lam)) <= l2:
                    if imu >= lam:
                        W[il,imu-lam,im1] = w3j[iwig]
                    iwig = iwig + 1
    cpm = np.zeros(nmu, dtype=np.complex128)
    cpm[lam] = c2r[lam,lam].real
    for mu in range(1, lam+1):
        for imu in (lam-mu, lam+mu):
            cpm[imu] = c2r[imu,lam+mu] + (1.0 - 2.0*(mu % 2)) * np.conj(c2r[imu,lam-mu])
    return W, cpm


@_njit_cached(fastmath = True)
def _atom_norm2(W, cpm, llvec, lam, v1, v2, iat, nang1, nang2, nrad1, nrad2):
    """sum of the squares of all features of atom iat, and the bound of _norm2;
    the sum is valid only where the bound is finite"""
    A1 = _density_matrix(v1, iat, nang1, nrad1)
    # v2 is a copy of v1 when rep1 == rep2
    if nang1 == nang2 and nrad1 == nrad2 and _same_atom(v1, v2, iat):
        A2 = A1
    else:
        A2 = _density_matrix(v2, iat, nang2, nrad2)
    inner, bound = _norm2(W, llvec, lam, A1, A2)
    # bound sums |v|^2 over every coefficient the features read; the terms of the
    # density-matrix sum are of size bound, and where they cancel to a small
    # fraction of it the features are summed directly
    if _finite(bound) and inner < 1e-3 * bound:
        inner = _norm2_direct(W, cpm, llvec, lam, v1, v2, iat, nrad1, nrad2)
    return inner, bound


@_njit_cached(parallel=True, fastmath = True)
def lam0_norm(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec):
    """|q0| of each atom: the length of its raw lambda = 0 vector over all
    nrad1*nrad2*llmax features, as equicombsparse_numba computes it for lambda = 0.
    w3j, llmax and llvec are those of lambda = 0.

    The equicomb kernels take it as norm0: with descriptor.normalisation
    rel_power, lambda > 0 is divided by norm0 in place of its own length;
    an empty norm0 is unit normalisation."""
    W, cpm = _wigner_tables(w3j, llmax, llvec, 0, np.ones((1, 1), dtype=np.complex128), nang1)
    norm0 = np.zeros(natoms, dtype=np.float64)
    for iat in prange(natoms):
        inner, bound = _atom_norm2(W, cpm, llvec, 0, v1, v2, iat, nang1, nang2, nrad1, nrad2)
        norm0[iat] = np.sqrt(inner)
    return norm0


@njit(parallel=True, fastmath = True)
def grad_lam0_norm(natoms,natoms_range,nrad1,nrad2,v1,v2,dv1,dv2,w3j,llmax,llvec):
    """|q0| of each atom as the gradient kernels compute it for lambda = 0, and
    dot0[iat, i_grad, x] = q0 . dq0 / dr_x(i_grad): the norm0 and dot0 arguments
    of grad_equicomb_numba and grad_equicombsparse_numba"""
    norm0 = np.zeros(natoms_range, dtype=np.float64)
    dot0 = np.zeros((natoms_range, natoms, 3), dtype=np.float64)
    for iat in prange(natoms_range):
        inner = 0.0
        dpc = np.zeros((natoms, 3), dtype=np.complex128)
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l = llvec[il,0]
                    pc = 0j
                    dpc[:] = 0.0
                    for im in range(2*l+1):
                        a = v1[iat,n1,l,im]
                        b = np.conj(v2[iat,n2,l,im])
                        pc += w3j[iwig] * a * b
                        for i_grad in range(natoms):
                            for x in range(3):
                                dpc[i_grad,x] += w3j[iwig] * (dv1[iat,n1,l,im,i_grad,x] * b + a * np.conj(dv2[iat,n2,l,im,i_grad,x]))
                        iwig = iwig + 1
                    inner = inner + pc.real**2
                    for i_grad in range(natoms):
                        for x in range(3):
                            dot0[iat,i_grad,x] += pc.real * dpc[i_grad,x].real
        norm0[iat] = np.sqrt(inner)
    return norm0, dot0


@_njit_cached(parallel=True, fastmath = True)
def equicombsparse_numba(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize,nfps,vfps,norm0):
    """The nfps features vfps of each atom's lam-SOAP vector, normalised over
    all featsize = nrad1*nrad2*llmax features.  Only the nfps features are built.
    With a non-empty norm0 (lam0_norm), lam > 0 is divided by norm0 instead.
    An atom whose features are all zero gets zero features; an atom with a NaN
    or inf coefficient gets NaN features.

    Requires v1 and v2 from real densities, v[n, l, -m] = (-1)^m conj(v[n, l, m]),
    llvec from get_angular_indexes_symmetric (l1 + l2 + lam even), w3j as written
    by salted/wigner.py, and c2r from complex_to_real_transformation.

    Feature ifeat = (n1*nrad2 + n2)*llmax + il is q = Re(c2r @ pc) with
    pc[mu] = sum_m1 w3j * v1[n1, l1, m1] * conj(v2[n2, l2, m1 - mu]).
    Then w3j[-mu, -m1] = w3j[mu, m1] and pc[-mu] = (-1)^mu conj(pc[mu]): only
    mu >= 0 is computed.  Row mu of c2r has entries in columns mu and -mu only,
    so q[mu] and q[-mu] are the real parts of two complex multiples of pc[|mu|].
    c2r @ pc is real and sum_mu q_mu^2 = |pc|^2 (c2r is unitary); summed over
    n1, n2 that factors into the per-l density matrices of _density_matrix.
    Where that sum cancels to a small fraction of its terms (features that
    nearly vanish by site symmetry), the norm is summed feature by feature.

    The m loops index with i0+t, j0+t (i0, j0, t >= 0), so each loop reads
    contiguous memory."""
    nmu = 2*lam+1
    W, cpm = _wigner_tables(w3j, llmax, llvec, lam, c2r, nang1)
    # the selected features grouped by il: features start[il]:start[il+1] have
    # channels sn1, sn2 and go to column scol of p
    start = np.zeros(llmax+1, dtype=np.int64)
    for n in range(nfps):
        start[vfps[n] % llmax + 1] += 1
    for il in range(llmax):
        start[il+1] += start[il]
    nextpos = start[:-1].copy()
    sn1 = np.empty(nfps, dtype=np.int64)
    sn2 = np.empty(nfps, dtype=np.int64)
    scol = np.empty(nfps, dtype=np.int64)
    for n in range(nfps):
        k = nextpos[vfps[n] % llmax]
        nextpos[vfps[n] % llmax] = k + 1
        sn1[k] = vfps[n] // (llmax * nrad2)
        sn2[k] = (vfps[n] // llmax) % nrad2
        scol[k] = n

    p = np.zeros((natoms, nmu, nfps), dtype=np.float64)
    for iat in prange(natoms):
        inner, bound = _atom_norm2(W, cpm, llvec, lam, v1, v2, iat, nang1, nang2, nrad1, nrad2)
        if not _finite(bound):
            p[iat,:,:] = np.nan
            continue
        normfact = 1 / np.sqrt(inner) if inner > 0.0 else 0.0
        if norm0.size > 0 and lam > 0:
            normfact = 1 / norm0[iat] if norm0[iat] > 0.0 else 0.0
        for il in range(llmax):
            l1, l2 = llvec[il,0], llvec[il,1]
            # mu = 0: the terms m1 and -m1 are complex conjugates and the m1 = 0 term
            # is real, so pc[0] = 2 Re(sum over m1 < 0) + the m1 = 0 term
            i0 = max(0, l1-l2)
            j0 = max(0, l2-l1)
            cnt = min(l1, l2)
            for k in range(start[il], start[il+1]):
                n1 = sn1[k]
                n2 = sn2[k]
                x = 0.0
                for t in range(cnt):
                    x += W[il,0,i0+t] * (v1[iat,n1,l1,i0+t] * np.conj(v2[iat,n2,l2,j0+t])).real
                x = 2*x + W[il,0,l1] * v1[iat,n1,l1,l1].real * v2[iat,n2,l2,l2].real
                p[iat,lam,scol[k]] = cpm[lam].real * x * normfact
            for mu in range(1, lam+1):
                # im1 = i0+t pairs with im2 = j0+t (m2 = m1-mu), t = 0..cnt-1
                i0 = max(0, l1+mu-l2)
                j0 = max(0, l2-l1-mu)
                cnt = min(2*l1-i0, 2*l2-j0)+1
                for k in range(start[il], start[il+1]):
                    n1 = sn1[k]
                    n2 = sn2[k]
                    s = 0j
                    for t in range(cnt):
                        s += W[il,mu,i0+t] * v1[iat,n1,l1,i0+t] * np.conj(v2[iat,n2,l2,j0+t])
                    p[iat,lam+mu,scol[k]] = (cpm[lam+mu] * s).real * normfact
                    p[iat,lam-mu,scol[k]] = (cpm[lam-mu] * s).real * normfact
    return p

@njit(parallel=True, fastmath = True)
def equicomb_numba(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize,norm0):
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
        if norm0.size > 0:
            # rel_power: lam > 0 is divided by |q0|; an atom without density keeps zeros
            if lam > 0:
                normfact = norm0[iat]
            if normfact == 0.0:
                continue
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu] / normfact
    return p

@njit(parallel=True, fastmath = True)
def equicombfps(natoms, nang1, nang2, nrad1, nrad2, v1, v2, w3j, llmax, llvec, lam, c2r, featsize, norm0):
    p = np.zeros((featsize, natoms * (2*lam+1)), dtype=np.float64)
    v2c  = np.conj(v2)
    for iat in prange(natoms):
        inner = 0.0
        ptemp = np.zeros((featsize,2*lam+1), dtype=np.float64)
        ifeat = 0
        for n1 in range(nrad1):
            for n2 in range(nrad2):
                iwig = 0
                for il in range(llmax):
                    l1 = llvec[il,0]
                    l2 = llvec[il,1]
                    pcmplx = np.zeros(2*lam+1, dtype=np.complex128)
                    for imu in range(2*lam+1):
                        mu = imu - lam
                        for im1 in range(2*l1+1):
                            m1 = im1 - l1
                            m2 = m1 - mu
                            if (abs(m2) <= l2):
                                im2 = m2 + l2
                                v2cv  = v2c[iat,n2,l2,im2]
                                v1v = v1[iat,n1,l1,im1]
                                pcmplx[imu] = pcmplx[imu] + w3j[iwig] * v1v * v2cv
                                iwig = iwig + 1
                    preal = np.zeros(2*lam+1, dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                            preal[imu] = preal[imu] + np.real(c2r[imu,im1] * pcmplx[im1])
                        inner = inner + preal[imu]**2
                        ptemp[ifeat, imu] = preal[imu]
                    ifeat = ifeat + 1
        normfact = np.sqrt(inner)
        if norm0.size > 0:
            # rel_power: lam > 0 is divided by |q0|; an atom without density keeps zeros
            if lam > 0:
                normfact = norm0[iat]
            if normfact == 0.0:
                continue
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[ifeat, iat*(2*lam+1) + imu] = ptemp[ifeat, imu] / normfact
    
    return p

@njit(parallel=True, fastmath = True)
def equicombnonorm(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize):
    p = np.zeros((natoms, 2*lam+1, featsize), dtype=np.float64)
    v2c  = np.conj(v2)

    for iat in prange(natoms):
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
                        ptemp[ifeat,imu] = preal[imu]
                    ifeat = ifeat + 1
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu]
    return p

@njit(parallel=True, fastmath = True)
def antiequicombnonorm(natoms,nang1,nang2,nrad1,nrad2,v1,v2,w3j,llmax,llvec,lam,c2r,featsize):
    p = np.zeros((natoms, 2*lam+1, featsize), dtype=np.float64)
    v2c  = np.conj(v2)

    for iat in prange(natoms):
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
                    pimag = np.zeros(2*lam+1, dtype=np.float64)
                    for imu in range(2*lam+1):
                        for im1 in range(2*lam+1):
                             pimag[imu] = pimag[imu] + np.imag(c2r[imu,im1] * pcmplx[im1])
                        ptemp[ifeat,imu] = pimag[imu]
                    ifeat = ifeat + 1
        for ifeat in range(featsize):
            for imu in range(2*lam+1):
                p[iat,imu,ifeat] = ptemp[ifeat,imu]
    return p

@njit(parallel=True, fastmath = True)
def kernelequicomb(n1,n2,lam1,lam2,L,nsize,msize,cgsize,cgcoefs,knm,k0):
    mu1size = 2*lam1+1
    mu2size = 2*lam2+1
    
    kernel = np.zeros((nsize,msize), dtype=np.complex128)
    iM1 = 0
    idx1 = 0
    for i1 in range(n1):
        icg1 = 0
        for imu1 in range(mu1size):
            mu1 = imu1-lam1
            for ik1 in range(mu2size):
                k1 = ik1-lam2
                M1 = mu1+k1
                if (abs(M1) <= L):
                    j1 = M1+L
                    cg1 = cgcoefs[icg1]
                    iM2 = 0
                    idx2 = 0
                    for i2 in range(n2):
                        icg2 = 0
                        for imu2 in range(mu1size):
                            mu2 = imu2-lam1
                            for ik2 in range(mu2size):
                                k2 = ik2-lam2
                                M2 = mu2+k2
                                if (abs(M2) <= L):
                                    j2 = M2+L
                                    cg2 = cgcoefs[icg2]
                                    kernel[iM1,iM2] = kernel[iM1,iM2] + cg1 * cg2 * knm[idx1+j1,idx2+j2] * k0[i1,i2]
                                    icg2 = icg2 + 1
                                iM2 = iM2 + 1
                        idx2 = idx2 + 2*L+1
                    icg1 = icg1 + 1
                iM1 = iM1 + 1
        idx1 = idx1 + 2*L+1
    return kernel

@njit(parallel=True, fastmath = True)
def kernelnorm(n1,n2,msize,normfact1,normfact2,kernel):
    
    knorm = np.zeros((n1*msize,n2*msize))
    j1 = 0
    for i1 in range(n1):
        for im1 in range(msize):
            j2 = 0
            for i2 in range(n2):
                for im2 in range(msize):
                    knorm[j1,j2] = kernel[j1,j2] / np.sqrt(normfact1[i1]*normfact2[i2])
                    j2 = j2 + 1
            j1 = j1 + 1
    
    return knorm
