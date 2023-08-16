import os
import sys
import ase
import time
import chemfiles
import numpy as np
from sympy.physics.wigner import wigner_3j
from ase.data import atomic_numbers

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from equistore import Labels

from lib import equicomb 
from lib import equicombfield 
import sph_utils
import efield
import basis

sys.path.insert(0, './')
import inp

filename = inp.filename
saltedname = inp.saltedname
predname = inp.predname
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
sparsify = inp.sparsify

from sys_utils import read_system, get_atom_idx
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

start = time.time()

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


########################################################################################


for iconf in range(ndata):
    # Define relevant species
    excluded_species = []
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))

# recompute number of atoms
natoms_total = 0
natoms_list = []
natoms = np.zeros(ndata,int)
for iconf in range(ndata):
    natoms[iconf] = 0
    for spe in species:
        natoms[iconf] += natom_dict[(iconf,spe)]
    natoms_total += natoms[iconf]
    natoms_list.append(natoms[iconf])
natoms_max = max(natoms_list)

# recompute atomic indexes from new species selections
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

#############################################################################

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


with chemfiles.Trajectory(filename) as trajectory:
    frames = [f for f in trajectory]

print(f"The dataset contains {len(frames)} frames.")

if rep1=="rho":
    # get SPH expansion for atomic density    
    calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

elif rep1=="V":
    # get SPH expansion for atomic potential 
    calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

else:
    print("Error: requested representation", rep1, "not provided")

nspe1 = len(neighspe1)
keys_array = np.zeros(((nang1+1)*len(species)*nspe1,3),int)
i = 0
for l in range(nang1+1):
    for specen in species:
        for speneigh in neighspe1:
            keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
            i += 1

keys_selection = Labels(
    names=["spherical_harmonics_l","species_center","species_neighbor"],
    values=keys_array
)

rhostart = time.time()

spx = calculator.compute(frames, selected_keys=keys_selection)
spx = spx.keys_to_properties("species_neighbor")
spx = spx.keys_to_samples("species_center")
 
# Get 1st set of coefficients as a complex numpy array
omega1 = np.zeros((nang1+1,natoms_total,2*nang1+1,nspe1*nrad1),complex)
for l in range(nang1+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)

print("rho time:", (time.time()-rhostart))

potstart = time.time()

# External field?
if inp.field:

    # get SPH expansion for a uniform and constant external field aligned along Z 
    omega2 = np.zeros((natoms_total,nrad2),complex)
    for iat in range(natoms_total):
        omega2[iat] = efield.get_efield_sph(nrad2,rcut2)

else:

    if rep2=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)
    
    elif rep2=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL) 
    
    else:
        print("Error: requested representation", rep2, "not provided")

    nspe2 = len(neighspe2)
    keys_array = np.zeros(((nang2+1)*len(species)*nspe2,3),int)
    i = 0
    for l in range(nang2+1):
        for specen in species:
            for speneigh in neighspe2:
                keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
                i+=1
    
    keys_selection = Labels(
        names=["spherical_harmonics_l","species_center","species_neighbor"],
        values=keys_array
    )
    
    spx_pot = calculator.compute(frames, selected_keys=keys_selection)
    spx_pot = spx_pot.keys_to_properties("species_neighbor")
    spx_pot = spx_pot.keys_to_samples("species_center")
   

    # Get 2nd set of coefficients as a complex numpy array 
    omega2 = np.zeros((nang2+1,natoms_total,2*nang2+1,nspe2*nrad2),complex)
    for l in range(nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)

print("pot time:", (time.time()-potstart))

# Generate directories for saving descriptors 
dirpath = os.path.join(inp.saltedpath, "equirepr_"+saltedname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
for lam in range(lmax_max+1):

    print("lambda =", lam)

    equistart = time.time()

    # External field?
    if inp.field:
        # Select relevant angular components for equivariant descriptor calculation
        llmax = 0
        lvalues = {}
        for l1 in range(nang1+1):
            # keep only even combination to enforce inversion symmetry
            if (lam+l1+1)%2==0 :
                if abs(1-lam) <= l1 and l1 <= (1+lam) :
                    lvalues[llmax] = [l1,1]
                    llmax+=1
    else:
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
    if inp.field:
        wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field.dat")
        wigdim = wigner3j.size 
    else:
        wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_lmax2-"+str(nang2)+".dat")
        wigdim = wigner3j.size
  
    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    v1 = np.transpose(omega1,(2,0,3,1))
    if inp.field:
        v2 = omega2.T
    else:
        v2 = np.transpose(omega2,(2,0,3,1))
    

    # Compute complex to real transformation matrix for the given lambda value
    c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

    if inp.field:
       # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018) by having the field components as second entry
       p = equicombfield.equicombfield(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
       # Define feature space size
       featsize = nspe1*nrad1*nrad2*llmax
    else:
       # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
       p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
       # Define feature space size 
       featsize = nspe1*nspe2*nrad1*nrad2*llmax

    # Reshape equivariant descriptor
    p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
 
    print("equivariant time:", (time.time()-equistart))
    
    normstart = time.time()
    
    # Normalize equivariant descriptor 
    inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsize),p.reshape(natoms_total,(2*lam+1)*featsize))
    p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
    print("norm time:", (time.time()-normstart))

    savestart = time.time()
   
    print("feature space size =", featsize)

    #TODO modify SALTED to directly deal with compact natoms_total dimension
    if lam==0:
        p = p.reshape(natoms_total,featsize)
        pvec = np.zeros((ndata,natoms_max,featsize))
    else:
        p = p.reshape(natoms_total,2*lam+1,featsize)
        pvec = np.zeros((ndata,natoms_max,2*lam+1,featsize))
    i = 0
    for iconf in range(ndata):
        for iat in range(natoms[iconf]):
            pvec[iconf,iat] = p[i]
            i += 1

    # Do feature selection with FPS sparsification
    if sparsify:
        if ncut==-1:
            print("ERROR: please select a finite number of features!")
        if ncut >= featsize:
            ncut = featsize + 1 - 1 
        print("fps...")
        pvec = pvec.reshape(ndata*natoms_max*(2*lam+1),featsize)
        vfps = do_fps(pvec.T,ncut,0)
        np.save(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy", vfps)

    # Apply sparsification with precomputed FPS selection 
    if not sparsify and ncut>-1:
        # Load sparsification details
        vfps = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy")
        print("sparsifying...")
        pvec = pvec.reshape(ndata*natoms_max*(2*lam+1),featsize)
        psparse = pvec.T[vfps].T
        if lam==0:
            psparse = psparse.reshape(ndata,natoms_max,psparse.shape[-1])
        else:
            psparse = psparse.reshape(ndata,natoms_max,(2*lam+1),psparse.shape[-1])
        # Save sparse feature vector
        if inp.field==True:
            np.save(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"_field.npy", psparse)
        else:
            np.save(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+".npy", psparse)
    
    # Save non-sparse descriptor  
    if not sparsify and ncut==-1:
        if inp.field==True:    
            np.save(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"_field.npy", pvec)
        else:
            np.save(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+".npy", pvec)
    
    print("save time:", (time.time()-savestart))

print("time:", (time.time()-start))
