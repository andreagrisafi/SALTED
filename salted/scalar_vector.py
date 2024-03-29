"""
TODO: replace class arraylist by numpy.concatenate
"""

import os
import sys
import time
import os.path as osp
from ase.io import read
import h5py

import numpy as np
from scipy import sparse
from ase.data import atomic_numbers

from salted.sys_utils import read_system,get_atom_idx,get_conf_range

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted import wigner
from salted import sph_utils
from salted import basis

from salted.lib import equicomb
from salted.lib import equicombsparse

def build():

    sys.path.insert(0, './')
    import inp

    # salted parameters
    filename = inp.filename
    saltedname = inp.saltedname
    sparsify = inp.sparsify
    rep1 = inp.rep1
    rcut1 = inp.rcut1
    sig1 = inp.sig1
    nrad1 = inp.nrad1
    nang1 = inp.nang1
    neighspe1 = inp.neighspe1
    rep2 = inp.rep2
    rcut2 = inp.rcut2
    sig2 = inp.sig2
    nrad2 = inp.nrad2
    nang2 = inp.nang2
    neighspe2 = inp.neighspe2
    ncut = inp.ncut
    M = inp.Menv
    zeta = inp.z

    sdir = osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}")

    if sparsify==False:
        # Generate directories for saving descriptors
        if not osp.exists(sdir):
            os.mkdir(sdir)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    HYPER_PARAMETERS_DENSITY = {
        "cutoff": rcut1,
        "max_radial": nrad1,
        "max_angular": nang1,
        "atomic_gaussian_width": sig1,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
        "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
    }

    HYPER_PARAMETERS_POTENTIAL = {
        "potential_exponent": 1,
        "cutoff": rcut2,
        "max_radial": nrad2,
        "max_angular": nang2,
        "atomic_gaussian_width": sig2,
        "center_atom_weight": 1.0,
        "radial_basis": {"Gto": {"spline_accuracy": 1e-6}}
    }

    if rep1=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep1=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        print("Error: requested representation", rep1, "not provided")


    # Load feature space sparsification information if required 
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                inp.saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    frames = read(filename,":")
    natoms_total = sum(natoms)
    conf_range = range(ndata)

    lam = 0

    # Select relevant angular components for equivariant descriptor calculation
    llmax = 0
    lvalues = {}
    for l1 in range(nang1+1):
        for l2 in range(nang2+1):
            # keep only even combination to enforce inversion symmetry
            if (lam+l1+l2)%2==0 :
                if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                    lvalues[llmax] = [l1,l2]
                    llmax+=1
    # Fill dense array from dictionary
    llvec = np.zeros((llmax,2),int)
    for il in range(llmax):
        llvec[il,0] = lvalues[il][0]
        llvec[il,1] = lvalues[il][1]

    if sparsify==False:
        wigner.build(False)

    # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
    wigner3j = np.loadtxt(os.path.join(
        inp.saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
    ))
    wigdim = wigner3j.size
    
    nspe1 = len(neighspe1)
    keys_array = np.zeros(((nang1+1)*len(species)*nspe1,4),int)
    i = 0
    for l in range(nang1+1):
        for specen in species:
            for speneigh in neighspe1:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx = calculator.compute(frames, selected_keys=keys_selection)
    spx = spx.keys_to_properties("neighbor_type")
    spx = spx.keys_to_samples("center_type")

    # Get 1st set of coefficients as a complex numpy array
    omega1 = np.zeros((nang1+1,natoms_total,2*nang1+1,nspe1*nrad1),complex)
    for l in range(nang1+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(o3_lambda=l).values)

    if rep2=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

    elif rep2=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

    else:
        print("Error: requested representation", rep2, "not provided")

    nspe2 = len(neighspe2)
    keys_array = np.zeros(((nang2+1)*len(species)*nspe2,4),int)
    i = 0
    for l in range(nang2+1):
        for specen in species:
            for speneigh in neighspe2:
                keys_array[i] = np.array([l,1,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i += 1

    keys_selection = Labels(
        names=["o3_lambda","o3_sigma","center_type","neighbor_type"],
        values=keys_array
    )

    spx_pot = calculator.compute(frames, selected_keys=keys_selection)
    spx_pot = spx_pot.keys_to_properties("neighbor_type")
    spx_pot = spx_pot.keys_to_samples("center_type")

    # Get 2nd set of coefficients as a complex numpy array 
    omega2 = np.zeros((nang2+1,natoms_total,2*nang2+1,nspe2*nrad2),complex)
    for l in range(nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(o3_lambda=l).values)

    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))
    
    # Compute complex to real transformation matrix for the given lambda value
    c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]
 
    start = time.time()    

    if sparsify:

        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        nfps = len(vfps[lam])
        p = equicombsparse.equicombsparse(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize,nfps,vfps[lam])
        p = np.transpose(p,(2,0,1))
        featsize = ncut

    else:

        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
        p = np.transpose(p,(2,0,1))
  
    print("time = ", time.time()-start)
 
    #TODO modify SALTED to directly deal with compact natoms_total dimension
    p = p.reshape(natoms_total,featsize)
    pvec = np.zeros((ndata,natmax,featsize))

    j = 0
    for i,iconf in enumerate(conf_range):
        for iat in range(natoms[iconf]):
            pvec[i,iat] = p[j]
            j += 1
 
    h5f = h5py.File(osp.join(sdir, f"FEAT-0.h5"), 'w')    
    h5f.create_dataset("descriptor",data=pvec)
    h5f.close()

if __name__ == "__main__":
    build()
