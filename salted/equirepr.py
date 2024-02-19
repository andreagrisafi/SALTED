import os
import sys
import time
from ase.data import atomic_numbers
from ase.io import read
import numpy as np
import h5py
import random

from rascaline import SphericalExpansion
from rascaline import LodeSphericalExpansion
from metatensor import Labels

from salted import wigner
from salted import sph_utils
from salted import efield
from salted import basis

from salted.lib import equicomb
from salted.lib import equicombfield
from salted.lib import equicombfieldx
from salted.lib import equicombfieldy

def build():

    sys.path.insert(0, './')
    import inp

    if inp.field==True and inp.vfield==True:
        if rank==0: print("ERROR: field and vfield cannot be run together!")
        sys.exit(0)

    if inp.sparsify:
        
        if inp.ncut > 0:

            if inp.field:
                equirepr(sparsify=True,field=True,vfield=False)

            if inp.vfield:
                equirepr(sparsify=True,field=False,vfield=True)

            equirepr(sparsify=True,field=False,vfield=False)

        else:

            if rank==0: print("ERROR: features cutoff ncut must be a positive integer!")
            sys.exit(0)
    
    else: 

        if inp.field:
            equirepr(sparsify=False,field=True,vfield=False)

        if inp.vfield:
            equirepr(sparsify=False,field=False,vfield=True)

        equirepr(sparsify=False,field=False,vfield=False)

    return


def equirepr(sparsify,field,vfield):
    sys.path.insert(0, './')
    import inp
    
    filename = inp.filename
    saltedname = inp.saltedname
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
    parallel = inp.parallel
    nsamples = inp.nsamples
    
    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
        size = 1
    
    from salted.sys_utils import read_system, get_atom_idx,get_conf_range
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    start = time.time()
   
    if nsamples < ndata and sparsify: ndata = inp.nsamples
    ndata_true = ndata
    if rank == 0: print(f"The dataset contains {ndata_true} frames.")
    
    if inp.parallel:
        if sparsify: 
            print("ERROR: sparsification cannot be run in parallel!")
            sys.exit(0)
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
        ndata = len(conf_range)
    else:
        if sparsify: 
            conf_range = list(range(ndata_true))
            random.Random(3).shuffle(list(range(ndata_true)))
            conf_range = conf_range[:ndata]
        else:
            conf_range = list(range(ndata))
    
    frames = read(filename,":")
    frames = [frames[i] for i in conf_range]
    natoms_total = sum(natoms[conf_range])
    
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
    
    if rank == 0: print("Repr. 1 time:", (time.time()-rhostart))
    
    potstart = time.time()
    
    # External field?
    if field or vfield:
    
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
    
    if rank == 0: print("Repr. 2 time:", (time.time()-potstart))
    
    # Generate directories for saving descriptors 
    dirpath = os.path.join(inp.saltedpath, "equirepr_"+saltedname)
    
    if rank == 0:
        wigner.build(field,vfield)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
    if size > 1: comm.Barrier()

    # Compute equivariant descriptors for each lambda value entering the SPH expansion of the electron density
    if field==False and inp.field==True: lmax_max=0
    if vfield==False and inp.vfield==True: lmax_max=0
    for lam in range(lmax_max+1):
    
        if rank == 0: print("lambda =", lam)
    
        #equistart = time.time()
    
        # External field?
        if field or vfield:
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
        if field:
            wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field.dat")
            wigdim = wigner3j.size
        elif vfield:
            wigner3j_x = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field-x.dat")
            wigner3j_y = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field-y.dat")
            wigner3j_z = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_field-z.dat")
            wigdim_x = wigner3j_x.size
            wigdim_y = wigner3j_y.size
            wigdim_z = wigner3j_z.size
        else:
            wigner3j = np.loadtxt(inp.saltedpath+"wigners/wigner_lam-"+str(lam)+"_lmax1-"+str(nang1)+"_lmax2-"+str(nang2)+".dat")
            wigdim = wigner3j.size
      
        # Reshape arrays of expansion coefficients for optimal Fortran indexing 
        v1 = np.transpose(omega1,(2,0,3,1))
        if field or vfield:
            v2 = omega2.T
        else:
            v2 = np.transpose(omega2,(2,0,3,1))
    
        # Compute complex to real transformation matrix for the given lambda value
        c2r = sph_utils.complex_to_real_transformation([2*lam+1])[0]
    
        if field:
            # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018) by having the field components as second entry
            p = equicombfield.equicombfield(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
            # Define feature space size
            featsize = nspe1*nrad1*nrad2*llmax
            p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)

        elif vfield:
            # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018) by having the field components as second entry
            px = equicombfieldx.equicombfieldx(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim_x,wigner3j_x,llmax,llvec.T,lam,c2r)
            py = equicombfieldy.equicombfieldy(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim_y,wigner3j_y,llmax,llvec.T,lam,c2r)
            pz = equicombfield.equicombfield(natoms_total,nang1,nspe1*nrad1,nrad2,v1,v2,wigdim_z,wigner3j_z,llmax,llvec.T,lam,c2r)
            # Define feature space size
            featsize = nspe1*nrad1*nrad2*llmax
            px = np.transpose(px,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
            py = np.transpose(py,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
            pz = np.transpose(pz,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
            
        else:
            # Perform symmetry-adapted combination following Eq.S19 of Grisafi et al., PRL 120, 036002 (2018)
            p = equicomb.equicomb(natoms_total,nang1,nang2,nspe1*nrad1,nspe2*nrad2,v1,v2,wigdim,wigner3j,llmax,llvec.T,lam,c2r)
            # Define feature space size 
            featsize = nspe1*nspe2*nrad1*nrad2*llmax
            p = np.transpose(p,(4,0,1,2,3)).reshape(natoms_total,2*lam+1,featsize)
    
        # Reshape equivariant descriptor
     
        #if rank == 0: print("equivariant time:", (time.time()-equistart))
        
        #normstart = time.time()
        
        # Normalize equivariant descriptor
        if vfield:
            inner = np.zeros(natoms_total)
            inner += np.einsum('ab,ab->a', px.reshape(natoms_total,(2*lam+1)*featsize),px.reshape(natoms_total,(2*lam+1)*featsize))
            inner += np.einsum('ab,ab->a', py.reshape(natoms_total,(2*lam+1)*featsize),py.reshape(natoms_total,(2*lam+1)*featsize))
            inner += np.einsum('ab,ab->a', pz.reshape(natoms_total,(2*lam+1)*featsize),pz.reshape(natoms_total,(2*lam+1)*featsize))
            px = np.einsum('abc,a->abc', px,1.0/np.sqrt(inner))
            py = np.einsum('abc,a->abc', py,1.0/np.sqrt(inner))
            pz = np.einsum('abc,a->abc', pz,1.0/np.sqrt(inner))
        else:
            inner = np.einsum('ab,ab->a', p.reshape(natoms_total,(2*lam+1)*featsize),p.reshape(natoms_total,(2*lam+1)*featsize))
            p = np.einsum('abc,a->abc', p,1.0/np.sqrt(inner))
    
        #if rank == 0: print("norm time:", (time.time()-normstart))
    
        #savestart = time.time()
       
        if rank == 0: print("feature space size =", featsize)
    
        #TODO modify SALTED to directly deal with compact natoms_total dimension
        if vfield:

            if lam==0:
                px = px.reshape(natoms_total,featsize)
                py = py.reshape(natoms_total,featsize)
                pz = pz.reshape(natoms_total,featsize)
                pvec_x = np.zeros((ndata,natmax,featsize))
                pvec_y = np.zeros((ndata,natmax,featsize))
                pvec_z = np.zeros((ndata,natmax,featsize))
            else:
                px = px.reshape(natoms_total,2*lam+1,featsize)
                py = py.reshape(natoms_total,2*lam+1,featsize)
                pz = pz.reshape(natoms_total,2*lam+1,featsize)
                pvec_x = np.zeros((ndata,natmax,2*lam+1,featsize))
                pvec_y = np.zeros((ndata,natmax,2*lam+1,featsize))
                pvec_z = np.zeros((ndata,natmax,2*lam+1,featsize))
   
            j = 0
            for i,iconf in enumerate(conf_range):
                for iat in range(natoms[iconf]):
                    pvec_x[i,iat] = px[j]
                    pvec_y[i,iat] = py[j]
                    pvec_z[i,iat] = pz[j]
                    j += 1
        else:

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
    
        # Do feature selection with FPS sparsification
        if sparsify:
            if ncut >= featsize:
                ncut = featsize  
            if rank == 0: print("fps...")
            if vfield:
                pvec_x = pvec_x.reshape(ndata*natmax*(2*lam+1),featsize)
                pvec_y = pvec_y.reshape(ndata*natmax*(2*lam+1),featsize)
                pvec_z = pvec_z.reshape(ndata*natmax*(2*lam+1),featsize)
                pvec = np.vstack((pvec_x,np.vstack((pvec_y,pvec_z))))
                vfps = do_fps(pvec.T,ncut,0)
                np.save(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_vfield.npy", vfps)
            else:
                pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
                vfps = do_fps(pvec.T,ncut,0)
                if field:
                    np.save(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_field.npy", vfps)
                else:
                    np.save(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy", vfps)
    
        else:
            if field==True:
                if inp.parallel:
                    h5f = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"_field.h5",'w',driver='mpio',comm=MPI.COMM_WORLD)
                else:
                    h5f = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"_field.h5",'w')
            elif vfield==True:
                if inp.parallel:
                    h5f_x = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-x.h5",'w',driver='mpio',comm=MPI.COMM_WORLD)
                    h5f_y = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-y.h5",'w',driver='mpio',comm=MPI.COMM_WORLD)
                    h5f_z = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-z.h5",'w',driver='mpio',comm=MPI.COMM_WORLD)
                else:
                    h5f_x = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-x.h5",'w')
                    h5f_y = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-y.h5",'w')
                    h5f_z = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+"-z.h5",'w')
            else:
                if inp.parallel:
                    h5f = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+".h5",'w',driver='mpio',comm=MPI.COMM_WORLD)
                else:
                    h5f = h5py.File(inp.saltedpath+"equirepr_"+saltedname+"/FEAT-"+str(lam)+".h5",'w')
    
            if ncut == 0 or ncut >= featsize:
                ncut_l = featsize
            else:
                ncut_l = ncut
            if vfield:
                if lam==0:
                    dset_x = h5f_x.create_dataset('descriptor_x',(ndata_true,natmax,ncut_l),dtype='float64')
                    dset_y = h5f_y.create_dataset('descriptor_y',(ndata_true,natmax,ncut_l),dtype='float64')
                    dset_z = h5f_z.create_dataset('descriptor_z',(ndata_true,natmax,ncut_l),dtype='float64')
                else:
                    dset_x = h5f_x.create_dataset("descriptor_x",(ndata_true,natmax,(2*lam+1),ncut_l),dtype='float64')
                    dset_y = h5f_y.create_dataset("descriptor_y",(ndata_true,natmax,(2*lam+1),ncut_l),dtype='float64')
                    dset_z = h5f_z.create_dataset("descriptor_z",(ndata_true,natmax,(2*lam+1),ncut_l),dtype='float64')
            else:
                if lam==0:
                    dset = h5f.create_dataset("descriptor",(ndata_true,natmax,ncut_l),dtype='float64')
                else:
                    dset = h5f.create_dataset("descriptor",(ndata_true,natmax,(2*lam+1),ncut_l),dtype='float64')
    
            # Apply sparsification with precomputed FPS selection 
            if ncut_l < featsize:
                # Load sparsification details
                try:
                    if field:
                        vfps = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_field.npy")
                    elif vfield:
                        vfps = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+"_vfield.npy")
                    else:
                        vfps = np.load(inp.saltedpath+"equirepr_"+saltedname+"/fps"+str(ncut)+"-"+str(lam)+".npy")
                except:
                    if rank == 0: print("Sparsification must be performed prior to calculating the descritors if ncut > 0.")
                    exit()
                if rank == 0: print("sparsifying...")

                if vfield:
                    pvec_x = pvec_x.reshape(ndata*natmax*(2*lam+1),featsize)
                    pvec_y = pvec_y.reshape(ndata*natmax*(2*lam+1),featsize)
                    pvec_z = pvec_z.reshape(ndata*natmax*(2*lam+1),featsize)
                    psparse_x = pvec_x.T[vfps].T
                    psparse_y = pvec_y.T[vfps].T
                    psparse_z = pvec_z.T[vfps].T
                    if lam==0:
                        psparse_x = psparse_x.reshape(ndata,natmax,psparse_x.shape[-1])
                        psparse_y = psparse_y.reshape(ndata,natmax,psparse_y.shape[-1])
                        psparse_z = psparse_z.reshape(ndata,natmax,psparse_z.shape[-1])
                    else:
                        psparse_x = psparse_x.reshape(ndata,natmax,(2*lam+1),psparse_x.shape[-1])
                        psparse_y = psparse_y.reshape(ndata,natmax,(2*lam+1),psparse_y.shape[-1])
                        psparse_z = psparse_z.reshape(ndata,natmax,(2*lam+1),psparse_z.shape[-1])
                    # Save sparse feature vector
                    dset_x[conf_range] = psparse_x
                    dset_y[conf_range] = psparse_y
                    dset_z[conf_range] = psparse_z
                else:
                    pvec = pvec.reshape(ndata*natmax*(2*lam+1),featsize)
                    psparse = pvec.T[vfps].T
                    if lam==0:
                        psparse = psparse.reshape(ndata,natmax,psparse.shape[-1])
                    else:
                        psparse = psparse.reshape(ndata,natmax,(2*lam+1),psparse.shape[-1])
                    # Save sparse feature vector
                    dset[conf_range] = psparse
    
            # Save non-sparse descriptor  
            else:
                if vfield:
                    dset_x[conf_range] = pvec_x
                    dset_y[conf_range] = pvec_y
                    dset_z[conf_range] = pvec_z
                    h5f_x.close()
                    h5f_y.close()
                    h5f_z.close()
                else:
                    dset[conf_range] = pvec
                    h5f.close()
    
    
        #if rank == 0: print("save time:", (time.time()-savestart))
    
    if rank == 0: print("time:", (time.time()-start))
    
    return

if __name__ == "__main__":
    build()
