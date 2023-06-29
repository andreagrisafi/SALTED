import os
import sys
import ase
import time
import chemfiles
import numpy as np
from ase.io import read
from scipy import special
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

from sys_utils import read_system, get_atom_idx
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

bohr2angs = 0.529177210670

# Kernel parameters 
M = inp.Menv
zeta = inp.z
eigcut = inp.eigcut
reg = inp.regul

if inp.qmcode=="cp2k":

    # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
    print("Reading auxiliary basis info...")
    alphas = {}
    sigmas = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            avals = np.loadtxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
            if nmax[(spe,l)]==1:
                alphas[(spe,l,0)] = float(avals)
                sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
            else:
                for n in range(nmax[(spe,l)]):
                    alphas[(spe,l,n)] = avals[n]
                    sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
    
    # compute integrals of basis functions (needed to a posteriori correction of the charge)
    charge_integrals = {}
    dipole_integrals = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            charge_integrals_temp = np.zeros(nmax[(spe,l)])
            dipole_integrals_temp = np.zeros(nmax[(spe,l)])
            for n in range(nmax[(spe,l)]):
                inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                dipole_radint = 2**float(1.0+float(l)/2.0) * sigmas[(spe,l,n)]**(4+l) * special.gamma(2.0+float(l)/2.0)
                charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
                dipole_integrals[(spe,l,n)] = dipole_radint * np.sqrt(4.0*np.pi/3.0) / np.sqrt(inner)

loadstart = time.time()

# Load sparse set of atomic environments 
fps_idx = np.loadtxt(inp.saltedpath+"equirepr_"+inp.saltedname+"/sparse_set_"+str(M)+".txt",int)[:,0]
fps_species = np.loadtxt(inp.saltedpath+"equirepr_"+inp.saltedname+"/sparse_set_"+str(M)+".txt",int)[:,1]
# divide sparse set per species
fps_indexes = {}
for spe in species:
    fps_indexes[spe] = []
for iref in range(M):
    fps_indexes[species[fps_species[iref]]].append(fps_idx[iref])
Mspe = {}
for spe in species:
    Mspe[spe] = len(fps_indexes[spe])

# Load training feature vectors and RKHS projection matrix 
pvec_train = {}
pvec_train_nofield = {}
power_env_sparse = {}
power_env_sparse_nofield = {}
Vmat = {}
for lam in range(lmax_max+1):
    pvec_train[lam] = np.load(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+"_field.npy")
    pvec_train_nofield[lam] = np.load(inp.saltedpath+"equirepr_"+inp.saltedname+"/FEAT-"+str(lam)+".npy")
    for spe in species:
        Vmat[(lam,spe)] = np.load(inp.saltedpath+"kernels_"+inp.saltedname+"/spe"+str(spe)+"_l"+str(lam)+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/projector.npy")
        if lam==0:
            power_env_sparse[(lam,spe)] = pvec_train[lam].reshape(pvec_train[lam].shape[0]*pvec_train[lam].shape[1],pvec_train[lam].shape[-1])[np.array(fps_indexes[spe],int)] 
            power_env_sparse_nofield[(lam,spe)] = pvec_train_nofield[lam].reshape(pvec_train_nofield[lam].shape[0]*pvec_train_nofield[lam].shape[1],pvec_train_nofield[lam].shape[-1])[np.array(fps_indexes[spe],int)] 
        else:
            power_env_sparse[(lam,spe)] = pvec_train[lam].reshape(pvec_train[lam].shape[0]*pvec_train[lam].shape[1],2*lam+1,pvec_train[lam].shape[-1])[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*lam+1),pvec_train[lam].shape[-1])
            power_env_sparse_nofield[(lam,spe)] = pvec_train_nofield[lam].reshape(pvec_train_nofield[lam].shape[0]*pvec_train_nofield[lam].shape[1],2*lam+1,pvec_train_nofield[lam].shape[-1])[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*lam+1),pvec_train_nofield[lam].shape[-1])

# load regression weights
ntrain = int(inp.Ntrain*inp.trainfrac)
weights = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+".npy")

print("load time:", (time.time()-loadstart))

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

for iconf in range(ndata):
    # Define excluded species
    excluded_species = []
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))

# recompute atomic indexes from new species selections
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

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

descstart = time.time()

nspe1 = len(inp.neighspe1)
keys_array = np.zeros(((inp.nang1+1)*len(species)*nspe1,3),int)
i = 0
for l in range(inp.nang1+1):
    for specen in species:
        for speneigh in inp.neighspe1:
            keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
            i += 1

keys_selection = Labels(
    names=["spherical_harmonics_l","species_center","species_neighbor"],
    values=keys_array
)

spx = calculator.compute(frames, selected_keys=keys_selection)
spx = spx.keys_to_properties("species_neighbor")
spx = spx.keys_to_samples("species_center")

# Get 1st set of coefficients as a complex numpy array
omega1 = np.zeros((inp.nang1+1,natoms_total,2*inp.nang1+1,nspe1*inp.nrad1),complex)
for l in range(inp.nang1+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)

if inp.rep2=="rho":
    # get SPH expansion for atomic density    
    calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

elif inp.rep2=="V":
    # get SPH expansion for atomic potential 
    calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL) 

else:
    print("Error: requested representation", inp.rep2, "not provided")

nspe2 = len(inp.neighspe2)
keys_array = np.zeros(((inp.nang2+1)*len(species)*nspe2,3),int)
i = 0
for l in range(inp.nang2+1):
    for specen in species:
        for speneigh in inp.neighspe2:
            keys_array[i] = np.array([l,atomic_numbers[specen],atomic_numbers[speneigh]],int)
            i += 1

keys_selection = Labels(
    names=["spherical_harmonics_l","species_center","species_neighbor"],
    values=keys_array
)

spx_pot = calculator.compute(frames, selected_keys=keys_selection)
spx_pot = spx_pot.keys_to_properties("species_neighbor")
spx_pot = spx_pot.keys_to_samples("species_center")

# Get 2nd set of coefficients as a complex numpy array 
omega2 = np.zeros((inp.nang2+1,natoms_total,2*inp.nang2+1,nspe2*inp.nrad2),complex)
for l in range(inp.nang2+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)

# get SPH expansion for a uniform and constant external field aligned along Z 
omega_field = np.zeros((natoms_total,inp.nrad2),complex)
for iat in range(natoms_total):
    omega_field[iat] = efield.get_efield_sph(inp.nrad2,inp.rcut2)

print("coefficients time:", (time.time()-descstart))
print("")

dirpath = os.path.join(inp.saltedpath,"predictions_"+inp.saltedname+"_"+inp.predname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/","M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
psi_nm = {}
for lam in range(lmax_max+1):

    print("lambda =", lam)

    equistart = time.time()

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
    wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(inp.nang1)+"_lmax2-"+str(inp.nang2)+".dat")
    wigdim = wigner3j.size
  
    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute complex to real transformation matrix for the given lambda value
    c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

    # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
    p = equicomb.equicomb(natoms_total,inp.nang1,inp.nang2,nspe1*inp.nrad1,nspe2*inp.nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)

    # Define feature space and reshape equivariant descriptor
    featspace = nspe1*nspe2*inp.nrad1*inp.nrad2*llmax
    p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featspace)
 
    print("equivariant time:", (time.time()-equistart))
    
    normstart = time.time()
    
    # Normalize equivariant descriptor  
    inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featspace),p.reshape(natoms_total,(2*lam+1)*featspace))
    p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
    print("norm time:", (time.time()-normstart))

    fillstart = time.time()

    # Fill vector of equivariant descriptor 
    if lam==0:
        p = p.reshape(natoms_total,featspace)
        pvec = np.zeros((ndata,natoms_max,featspace))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec[iconf,iat] = p[i]
                i += 1
    else:
        pvec = np.zeros((ndata,natoms_max,2*lam+1,featspace))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec[iconf,iat] = p[i]
                i += 1

    print("fill vector time:", (time.time()-fillstart))

    #########################################################
    #                 START E-FIELD HERE
    #########################################################
 
    # Select relevant angular components for equivariant descriptor calculation
    llmax = 0
    lvalues = {}
    for l1 in range(inp.nang1+1):
        # keep only even combination to enforce inversion symmetry
        if (lam+l1+1)%2==0 :
            if abs(1-lam) <= l1 and l1 <= (1+lam) :
                lvalues[llmax] = [l1,1]
                llmax+=1
    # Fill dense array from dictionary
    llvec = np.zeros((llmax,2),int)
    for il in range(llmax):
        llvec[il,0] = lvalues[il][0]
        llvec[il,1] = lvalues[il][1]

    # Load the relevant Wigner-3J symbols associated with the given triplet (lam, lmax1, lmax2)
    wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(inp.nang1)+"_field.dat")
    wigdim = wigner3j.size
 
    # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
    v2 = omega_field.T
    p = equicombfield.equicombfield(natoms_total,inp.nang1,nspe1*inp.nrad1,inp.nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)

    # Define feature space and reshape equivariant descriptor
    featspacefield = nspe1*inp.nrad1*inp.nrad2*llmax
    p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featspacefield)
  
    print("equivariant time:", (time.time()-equistart))
     
    normstart = time.time()

    # Normalize equivariant descriptor  
    inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featspacefield),p.reshape(natoms_total,(2*lam+1)*featspacefield))
    p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))

    print("norm time:", (time.time()-normstart))

    fillstart = time.time()

    # Fill vector of equivariant descriptor 
    if lam==0:
        p = p.reshape(natoms_total,featspacefield)
        pvec_field = np.zeros((ndata,natoms_max,featspacefield))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec_field[iconf,iat] = p[i]
                i += 1
    else:
        pvec_field = np.zeros((ndata,natoms_max,2*lam+1,featspacefield))
        i = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec_field[iconf,iat] = p[i]
                i += 1
    
    print("fill vector time:", (time.time()-fillstart))

    rkhsstart = time.time()

    if lam==0:

        # Compute scalar kernels
        kernel0_nm = {}
        for iconf in range(ndata):
            for spe in species:
                kernel_nm = np.dot(pvec_field[iconf,atom_idx[(iconf,spe)]],power_env_sparse[(lam,spe)].T)
                kernel0_nm[(iconf,spe)] = np.dot(pvec[iconf,atom_idx[(iconf,spe)]],power_env_sparse_nofield[(lam,spe)].T)
                kernel_nm += kernel0_nm[(iconf,spe)]
                kernel_nm *= kernel0_nm[(iconf,spe)]**(zeta-1)
                # Project on RKHS
                psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])    
    else:

        # Compute covariant kernels
        for iconf in range(ndata):
            for spe in species:
                kernel_nm = np.dot(pvec_field[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse[(lam,spe)].T)
                kernel_nm += np.dot(pvec[iconf,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),pvec.shape[-1]),power_env_sparse_nofield[(lam,spe)].T)
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(Mspe[spe]):
                        kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                # Project on RKHS
                psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])    
    
    print("rkhs time:", time.time()-rkhsstart)

predstart = time.time()

if inp.qmcode=="cp2k":
    xyzfile = read(inp.filename,":")
    qfile = open(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+"/charges.dat","w")
    dfile = open(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+"/dipoles.dat","w")

# Load spherical averages if required
if inp.average:
    av_coefs = {}
    for spe in species:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# Perform equivariant predictions
for iconf in range(ndata):

    Tsize = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Tsize += 2*l+1

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = psi_nm[(iconf,spe,l)].shape[1]
                C[(spe,l,n)] = np.dot(psi_nm[(iconf,spe,l)],weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1

    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    if inp.average:
        Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                if inp.average and l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # add back spherical averages if required
    if inp.average:
        pred_coefs += Av_coeffs

    if inp.qmcode=="cp2k":

        # get geometry ingormation for dipole calculation
        geom = xyzfile[iconf]
        geom.wrap()
        coords = geom.get_positions()/bohr2angs
        all_symbols = xyzfile[iconf].get_chemical_symbols()
        all_natoms = int(len(all_symbols))

        if inp.average:
            # compute integral of predicted density
            iaux = 0
            rho_int = 0.0
            nele = 0.0
            for iat in range(all_natoms):
                spe = all_symbols[iat]
                if spe in species:
                    nele += inp.pseudocharge
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                            iaux += 2*l+1

        # compute charge and dipole 
        iaux = 0
        charge = 0.0
        dipole = 0.0
        for iat in range(all_natoms):
            spe = all_symbols[iat]
            if spe in species:
                if inp.average:
                    dipole += inp.pseudocharge * coords[iat,2]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        for im in range(2*l+1):
                            if l==0 and im==0:
                                if inp.average:
                                    pred_coefs[iaux] *= nele/rho_int
                                charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                                dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                            if l==1 and im==1:
                                dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                            iaux += 1
        print(iconf+1,dipole,file=dfile)
        if inp.average:
            print(iconf+1,rho_int,file=qfile)
        else:
            print(iconf+1,charge,file=qfile)

        # save predicted coefficients in CP2K format
        np.savetxt(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs)

    # save predicted coefficients
    np.save(inp.saltedpath+"predictions_"+inp.saltedname+"_"+inp.predname+"/M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(inp.regul)))+"/prediction_conf"+str(iconf)+".npy",pred_coefs)


if inp.qmcode=="cp2k":
    qfile.close()
    dfile.close()

print("")
print("prediction time:", time.time()-predstart)

print("")
print("total time:", (time.time()-start))
