import os
import sys
import time
import h5py
import chemfiles
import numpy as np
from scipy import special
from ase.data import atomic_numbers

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from equistore import Labels

import pathlib
SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../"
sys.path.append(SALTEDPATHLIB)
from lib import equicomb 
from lib import equicombfield 

import sph_utils
import basis

sys.path.insert(0, './')
import inp

if inp.parallel:
    from mpi4py import MPI
    # MPI information
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
#    print('This is task',rank+1,'of',size)
else:
    rank = 0

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

from sys_utils import read_system, get_atom_idx,get_conf_range
species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename)
atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

bohr2angs = 0.529177210670

# Kernel parameters 
M = inp.Menv
zeta = inp.z
reg = inp.regul

# Distribute structures to tasks
ndata_true = ndata
if inp.parallel:
    conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
    conf_range = comm.scatter(conf_range,root=0)
    ndata = len(conf_range)
    print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
else:
    conf_range = list(range(ndata))
natoms_total = sum(natoms[conf_range])

if inp.qmcode=="cp2k":

    # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
    if rank == 0: print("Reading auxiliary basis info...")
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

# Load training feature vectors and RKHS projection matrix 
Mspe = {}
power_env_sparse = {}
if inp.field: power_env_sparse_field = {}
Vmat = {}
vfps = {}
if inp.field: vfps_field = {}
for lam in range(lmax_max+1):
    # Load sparsification details
    if ncut > -1: vfps[lam] = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy")
    if ncut > -1 and inp.field: vfps_field[lam] = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_field.npy")
    for spe in species:
        power_env_sparse[(lam,spe)] = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-M-"+str(M)+".h5",'r')[spe][:]
        if inp.field: power_env_sparse_field[(lam,spe)] = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-M-"+str(M)+"_field.h5",'r')[spe][:]
        if lam == 0: Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
        if inp.field:
            Vmat[(lam,spe)] = np.load(inp.saltedpath+"kernels_"+saltedname+"_field/spe"+str(spe)+"_l"+str(lam)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")
        else:
            Vmat[(lam,spe)] = np.load(inp.saltedpath+"kernels_"+saltedname+"/spe"+str(spe)+"_l"+str(lam)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")

# load regression weights
ntrain = int(inp.Ntrain*inp.trainfrac)
if inp.field:
    weights = np.load(inp.saltedpath+"regrdir_"+saltedname+"_field/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
else:
    weights = np.load(inp.saltedpath+"regrdir_"+saltedname+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")

if rank == 0: print("load time:", (time.time()-loadstart))

start = time.time()

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
    frames = [frames[i] for i in conf_range]

if rank == 0: print(f"The dataset contains {ndata_true} frames.")

if rep1=="rho":
    # get SPH expansion for atomic density    
    calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

elif rep1=="V":
    # get SPH expansion for atomic potential 
    calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL)

else:
    if rank == 0: print("Error: requested representation", rep1, "not provided")

descstart = time.time()

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

spx = calculator.compute(frames, selected_keys=keys_selection)
spx = spx.keys_to_properties("species_neighbor")
spx = spx.keys_to_samples("species_center")

# Get 1st set of coefficients as a complex numpy array
omega1 = np.zeros((nang1+1,natoms_total,2*nang1+1,nspe1*nrad1),complex)
for l in range(nang1+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega1[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx.block(spherical_harmonics_l=l).values)

if rep2=="rho":
    # get SPH expansion for atomic density    
    calculator = SphericalExpansion(**HYPER_PARAMETERS_DENSITY)

elif rep2=="V":
    # get SPH expansion for atomic potential 
    calculator = LodeSphericalExpansion(**HYPER_PARAMETERS_POTENTIAL) 

else:
    if rank == 0: print("Error: requested representation", rep2, "not provided")

nspe2 = len(neighspe2)
keys_array = np.zeros(((nang2+1)*len(species)*nspe2,3),int)
i = 0
for l in range(nang2+1):
    for specen in species:
        for speneigh in neighspe2:
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
omega2 = np.zeros((nang2+1,natoms_total,2*nang2+1,nspe2*nrad2),complex)
for l in range(nang2+1):
    c2r = sph_utils.complex_to_real_transformation([2*l+1])[0]
    omega2[l,:,:2*l+1,:] = np.einsum('cr,ard->acd',np.conj(c2r.T),spx_pot.block(spherical_harmonics_l=l).values)

if inp.field:
# get SPH expansion for a uniform and constant external field aligned along Z 
    omega_field = np.zeros((natoms_total,nrad2),complex)
    for iat in range(natoms_total):
        omega_field[iat] = efield.get_efield_sph(nrad2,rcut2)

if rank == 0: print("coefficients time:", (time.time()-descstart))
if rank == 0: print("")

dirpath = os.path.join(inp.saltedpath,"predictions_"+saltedname)
if inp.field: dirpath += '_field'
dirpath += '_'+predname
if rank == 0 and not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(dirpath,"M"+str(M)+"_zeta"+str(zeta))
if rank == 0 and not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(dirpath,"N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if rank == 0 and not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
psi_nm = {}
for lam in range(lmax_max+1):

    if rank == 0: print("lambda =", lam)

    equistart = time.time()

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
    wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_lmax2-"+str(nang2)+".dat")
    wigdim = wigner3j.size
  
    # Reshape arrays of expansion coefficients for optimal Fortran indexing 
    v1 = np.transpose(omega1,(2,0,3,1))
    v2 = np.transpose(omega2,(2,0,3,1))

    # Compute complex to real transformation matrix for the given lambda value
    c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]

    # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
    p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)

    # Define feature space and reshape equivariant descriptor
    featsize = nspe1*nspe2*nrad1*nrad2*llmax
    p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
 
    if rank == 0: print("equivariant time:", (time.time()-equistart))
    
    normstart = time.time()
    
    # Normalize equivariant descriptor  
    inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsize),p.reshape(natoms_total,(2*lam+1)*featsize))
    p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
    if rank == 0: print("norm time:", (time.time()-normstart))

    sparsestart = time.time()
    
    if ncut > -1:
        p = p.reshape(natoms_total*(2*lam+1),featsize)
        p = p.T[vfps[lam]].T
        featsize = inp.ncut
    
    if rank == 0: print("sparse time:", (time.time()-sparsestart))
    
    fillstart = time.time()

    # Fill vector of equivariant descriptor 
    if lam==0:
        p = p.reshape(natoms_total,featsize)
        pvec = np.zeros((ndata,natmax,featsize))
    else:
        p = p.reshape(natoms_total,2*lam+1,featsize)
        pvec = np.zeros((ndata,natmax,2*lam+1,featsize))

    j = 0
    for i,iconf in enumerate(conf_range):
        for iat in range(natoms[iconf]):
            pvec[i,iat] = p[j]
            j += 1
    
    if rank == 0: print("fill vector time:", (time.time()-fillstart))

    if inp.field:
         #########################################################
         #                 START E-FIELD HERE
         #########################################################
      
         # Select relevant angular components for equivariant descriptor calculation
         llmax = 0
         lvalues = {}
         for l1 in range(nang1+1):
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
         wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field.dat")
         wigdim = wigner3j.size
      
         # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
         v2 = omega_field.T
         p = equicombfield.equicombfield(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
 
         # Define feature space and reshape equivariant descriptor
         featsizefield = nspe1*nrad1*nrad2*llmax
         p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsizefield)
       
         if rank == 0: print("field equivariant time:", (time.time()-equistart))
          
         normstart = time.time()
 
         # Normalize equivariant descriptor  
         inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsizefield),p.reshape(natoms_total,(2*lam+1)*featsizefield))
         p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
         if rank == 0: print("field norm time:", (time.time()-normstart))

         if ncut > -1:
             p = p.reshape(natoms_total*(2*lam+1),featsizefield)
             p = p.T[vfps_field[lam]].T
             featsizefield = inp.ncut 

         fillstart = time.time()
 
         # Fill vector of equivariant descriptor 
         if lam==0:
             p = p.reshape(natoms_total,featsizefield)
             pvec_field = np.zeros((ndata,natmax,featsizefield))
         else:
             p = p.reshape(natoms_total,2*lam+1,featsizefield)
             pvec_field = np.zeros((ndata,natmax,2*lam+1,featsizefield))

         j = 0
         for i,iconf in enumerate(conf_range):
             for iat in range(natoms[iconf]):
                 pvec_field[i,iat] = p[j]
                 j += 1
         
#         if ncut > -1:
#             pvec_field = pvec_field.T[vfps_field[lam]].T

    rkhsstart = time.time()

    if lam==0:

        # Compute scalar kernels
        kernel0_nm = {}
        for i,iconf in enumerate(conf_range):
            for spe in species:
                kernel0_nm[(iconf,spe)] = np.dot(pvec[i,atom_idx[(iconf,spe)]],power_env_sparse[(lam,spe)].T)
                if inp.field:
                    kernel_nm = np.dot(pvec_field[i,atom_idx[(iconf,spe)]],power_env_sparse_field[(lam,spe)].T) + kernel0_nm[(iconf,spe)]
                    kernel_nm *= kernel0_nm[(iconf,spe)]**(zeta-1)
                else:
                    kernel_nm = kernel0_nm[(iconf,spe)]**zeta
                # Project on RKHS
                psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])    
    else:

        # Compute covariant kernels
        for i,iconf in enumerate(conf_range):
            for spe in species:
                kernel_nm = np.dot(pvec[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),featsize),power_env_sparse[(lam,spe)].T)
                if inp.field: kernel_nm += np.dot(pvec_field[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*lam+1),pvec_field.shape[-1]),power_env_sparse_field[(lam,spe)].T)
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(Mspe[spe]):
                        kernel_nm[i1*(2*lam+1):i1*(2*lam+1)+2*lam+1][:,i2*(2*lam+1):i2*(2*lam+1)+2*lam+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                # Project on RKHS
                psi_nm[(iconf,spe,lam)] = np.dot(kernel_nm,Vmat[(lam,spe)])

    if rank == 0: print("rkhs time:", time.time()-rkhsstart,flush=True)

predstart = time.time()

if inp.qmcode=="cp2k":
    from ase.io import read
    xyzfile = read(filename,":")
    if rank == 0 and os.path.exists(dirpath+"/charges.dat"): os.remove(dirpath+"/charges.dat")
    if rank == 0 and os.path.exists(dirpath+"/dipoles.dat"): os.remove(dirpath+"/dipoles.dat")
    if inp.parallel: comm.Barrier()
    qfile = open(dirpath+"/charges.dat","a")
    dfile = open(dirpath+"/dipoles.dat","a")

# Load spherical averages if required
if inp.average:
    av_coefs = {}
    for spe in species:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# Perform equivariant predictions
for iconf in conf_range:

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
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                Mcut = psi_nm[(iconf,spe,l)].shape[1]
                C[(spe,l,n)] = np.dot(psi_nm[(iconf,spe,l)],weights[isize:isize+Mcut])
                isize += Mcut

    # init averages array if asked
    if inp.average:
        Av_coeffs = np.zeros(Tsize)
    
    # fill vector of predictions
    i = 0
    pred_coefs = np.zeros(Tsize)
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
        cell = geom.get_cell()/bohr2angs
        all_symbols = xyzfile[iconf].get_chemical_symbols()
        all_natoms = int(len(all_symbols))

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
        charge_right = 0.0
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
                               else:
                                   if n==nmax[(spe,l)]-1:
                                       pred_coefs[iaux] -= rho_int/(charge_integrals[(spe,l,n)]*natoms[iconf])
                               charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                               dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                           if l==1 and im==1:
                               dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                           iaux += 1
        print(iconf+1,dipole,file=dfile)
        print(iconf+1,rho_int,charge,file=qfile)

        # save predicted coefficients in CP2K format
        np.savetxt(dirpath+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs)

    # save predicted coefficients
    np.save(dirpath+"/prediction_conf"+str(iconf)+".npy",pred_coefs)


if inp.qmcode=="cp2k":
    qfile.close()
    dfile.close()
    if inp.parallel and rank == 0:
        dips = np.loadtxt(dirpath+'/dipoles.dat')
        np.savetxt(dirpath+'/dipoles.dat',dips[dips[:,0].argsort()],fmt='%i %f')
        qs = np.loadtxt(dirpath+'/charges.dat')
        np.savetxt(dirpath+'/charges.dat',qs[qs[:,0].argsort()],fmt='%i %f')

if rank == 0: print("")
if rank == 0: print("prediction time:", time.time()-predstart)

if rank == 0: print("")
if rank == 0: print("total time:", (time.time()-start))
