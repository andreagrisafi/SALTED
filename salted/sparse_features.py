import os
import random
import sys
import time
import os.path as osp

import numpy as np
import h5py
from ase.data import atomic_numbers
from ase.io import read
from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted import wigner
from salted import sph_utils
from salted import basis

from salted.lib import equicomb
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range



def build():
    inp = ParseConfig().parse_input()
    (saltedname, saltedpath,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    z, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()


    # Generate directories for saving descriptors
    sdir = osp.join(saltedpath, f"equirepr_{saltedname}")
    if not osp.exists(sdir):
        os.mkdir(sdir)

    if ncut <= 0:
        print("ERROR: features cutoff ncut must be a positive integer!")
        sys.exit(0)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    start = time.time()

    if nsamples <= ndata:
        ndata = nsamples
    else:
        print("ERROR: features cutoff ncut must be a positive integer!")
        sys.exit(0)

    ndata_true = ndata
    print(f"The dataset contains {ndata_true} frames.")

    conf_range = list(range(ndata_true))
    random.Random(3).shuffle(conf_range)

    if nsamples <= ndata:
        ndata = nsamples
    else:
        print("ERROR: nsamples cannot be greater than ndata!")
        sys.exit(0)

    conf_range = conf_range[:ndata]
    print(f"Selected {ndata} frames.")

    frames = read(filename,":")
    frames = list( frames[i] for i in conf_range )
    natoms = list( natoms[i] for i in conf_range )
    natoms_total = sum(natoms)

    def do_fps(x, d=0, initial=-1):
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
                i+=1

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

    # compute Wigner-3j symbols
    wigner.build(False)

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    for lam in range(lmax_max+1):

        print("lambda =", lam)

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

        # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
        wigner3j = np.loadtxt(osp.join(saltedpath, "wigners", f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"))
        wigdim = wigner3j.size

        # Reshape arrays of expansion coefficients for optimal Fortran indexing
        v1 = np.transpose(omega1,(2,0,3,1))
        v2 = np.transpose(omega2,(2,0,3,1))

        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

        # compute normalized equivariant descriptor
        featsize = nspe1*nspe2*nrad1*nrad2*llmax
        p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r,featsize)
        p = np.transpose(p,(2,0,1))

        print(f"feature space size = {featsize}")

        #TODO modify SALTED to directly deal with compact natoms_total dimension
        if lam==0:
            p = p.reshape(natoms_total,featsize)
            pvec = np.zeros((ndata,natmax,featsize))
        else:
            p = p.reshape(natoms_total,2*lam+1,featsize)
            pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

        j = 0
        for i in range(ndata):
            for iat in range(natoms[i]):
                pvec[i,iat] = p[j]
                j += 1

        # Do feature selection with FPS sparsification
        if ncut >= featsize:
            ncut = featsize

        print("fps...")
        pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
        vfps = do_fps(pvec.T,ncut,0)
        np.save(osp.join(sdir, f"fps{ncut}-{lam}.npy"), vfps)

    print(f"time: {(time.time()-start):.2f}")


if __name__ == "__main__":
    build()
