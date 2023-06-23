import os
import sys
import ase
import time
import chemfiles
import numpy as np
from sympy.physics.wigner import wigner_3j

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

from sys_utils import read_system, get_atom_idx
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

start = time.time()

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

centers_selection = np.zeros((natoms_total,2),int)
i=0
for iconf in range(ndata):
    # Define selected centers 
    for spe in species:
        for iat in atom_idx[(iconf,spe)]:
            centers_selection[i,0] = iconf
            centers_selection[i,1] = iat 
            i+=1
#    # Define excluded species
#    excluded_species = []
#    for iat in range(natoms[iconf]):
#        spe = atomic_symbols[iconf][iat]
#        if spe not in species:
#            excluded_species.append(spe)
#    excluded_species = set(excluded_species)
#    for spe in excluded_species:
#        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))
#
## recompute atomic indexes from new species selections
#atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

HYPER_PARAMETERS_DENSITY = {
    "cutoff": inp.rcut1,
    "max_radial": inp.nrad1,
    "max_angular": inp.nang1,
    "atomic_gaussian_width": inp.sig1,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-6}},
    "cutoff_function": {"ShiftedCosine": {"width": 0.1}},
}

HYPER_PARAMETERS_POTENTIAL = {
    "potential_exponent": 1,
    "cutoff": inp.rcut2,
    "max_radial": inp.nrad2,
    "max_angular": inp.nang2,
    "atomic_gaussian_width": inp.sig2,
    "center_atom_weight": 1.0,
    "radial_basis": {"Gto": {"spline_accuracy": 1e-6}}
}


with chemfiles.Trajectory(inp.filename) as trajectory:
    frames = [f for f in trajectory]

print(f"The dataset contains {len(frames)} frames.")

if inp.rep1=="rho":
    # get SPH expansion for atomic density    
    calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

elif inp.rep1=="V":
    # get SPH expansion for atomic potential 
    calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

else:
    print("Error: requested representation", inp.rep1, "not provided")

selection = Labels(
    names=["structure", "center"],
    values=centers_selection,
)

rhostart = time.time()

spx = calculator.compute(frames, selected_samples=selection)
spx = spx.keys_to_properties("species_neighbor")
spx = spx.keys_to_samples("species_center")

# Get 1st set of coefficients as a complex numpy array
omega1 = np.zeros((inp.nang1+1,natoms_total,2*inp.nang1+1,inp.nspe1*inp.nrad1),complex)
for l in range(inp.nang1+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)

print("rho time:", (time.time()-rhostart))

potstart = time.time()

# External field?
if inp.field:

    # get SPH expansion for a uniform and constant external field aligned along Z 
    omega2 = np.zeros((natoms_total,inp.nrad2),complex)
    for iat in range(natoms_total):
        omega2[iat] = efield.get_efield_sph(inp.nrad2,inp.rcut2)

else:

    if inp.rep2=="rho":
        # get SPH expansion for atomic density    
        calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)
    
    elif inp.rep2=="V":
        # get SPH expansion for atomic potential 
        calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL) 
    
    else:
        print("Error: requested representation", inp.rep2, "not provided")
    
    spx_pot = calculator.compute(frames, selected_samples=selection)
    spx_pot = spx_pot.keys_to_properties("species_neighbor")
    spx_pot = spx_pot.keys_to_samples("species_center")
    
    # Get 2nd set of coefficients as a complex numpy array 
    omega2 = np.zeros((inp.nang2+1,natoms_total,2*inp.nang2+1,inp.nspe2*inp.nrad2),complex)
    for l in range(inp.nang2+1):
        c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
        omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)

print("pot time:", (time.time()-potstart))

# Generate directories for saving descriptors 
dirpath = os.path.join(inp.saltedpath, "equirepr_"+inp.saltedname)
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
        for l1 in range(inp.nang1+1):
            # keep only even combination to enforce inversion symmetry
            if (lam+l1+1)%2==0 :
                if abs(1-lam) <= l1 and l1 <= (1+lam) :
                    lvalues[llmax] = [l1,1]
                    llmax+=1
    else:
        # Select relevant angular components for equivariant descriptor calculation
        llmax = 0
        lvalues = {}
        for l1 in range(inp.nang1+1):
            for l2 in range(inp.nang2+1):
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
        wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(inp.nang1)+"_field.dat")
        wigdim = wigner3j.size 
    else:
        wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(inp.nang1)+"_lmax2-"+str(inp.nang2)+".dat")
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
       p = equicombfield.equicombfield(natoms_total,inp.nang1,inp.nspe1*inp.nrad1,inp.nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
       # Define feature space size
       featspace = inp.nspe1*inp.nrad1*inp.nrad2*llmax
    else:
       # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
       p = equicomb.equicomb(natoms_total,inp.nang1,inp.nang2,inp.nspe1*inp.nrad1,inp.nspe2*inp.nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
       # Define feature space size 
       featspace = inp.nspe1*inp.nspe2*inp.nrad1*inp.nrad2*llmax

    # Reshape equivariant descriptor
    p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featspace)
 
    print("equivariant time:", (time.time()-equistart))
    
    normstart = time.time()
    
    # Normalize equivariant descriptor 
    inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featspace),p.reshape(natoms_total,(2*lam+1)*featspace))
    p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
    print("norm time:", (time.time()-normstart))

    savestart = time.time()
    
    #TODO modify SALTED to directly deal with compact natoms_total dimension
    if lam==0:
        p = p.reshape(natoms_total,featspace)
        pvec = np.zeros((ndata,natoms_max,featspace))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec[iconf,iat] = p[i]
                i += 1
            #for spe in species:
            #    np.save(inp.saltedpath+inp.equidir+"spe"+str(spe)+"_l"+str(lam)+"/pvec_conf"+str(iconf)+".npy",pvec[iconf,atom_idx[(iconf,spe)]])
        if inp.field==True:    
            np.save(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+"_field.npy", pvec)
        else:
            np.save(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+".npy", pvec)
    else:
        pvec = np.zeros((ndata,natoms_max,2*lam+1,featspace))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec[iconf,iat] = p[i]
                i += 1
            #for spe in species:
            #    np.save(inp.saltedpath+inp.equidir+"spe"+str(spe)+"_l"+str(lam)+"/pvec_conf"+str(iconf)+".npy",pvec[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featspace))
        if inp.field==True:    
            np.save(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+"_field.npy", pvec)
        else:
            np.save(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+".npy", pvec)
    
    print("save time:", (time.time()-savestart))

print("time:", (time.time()-start))
