import os
import sys
import time
import os.path as osp

import math
import numpy as np
from scipy import special
from scipy import sparse
from itertools import islice
from scipy.interpolate import interp1d

#from sympy.parsing import mathematica
#from sympy import symbols
#from sympy import lambdify

from salted import basis
from salted.sys_utils import ParseConfig, read_system, get_atom_idx, get_conf_range

def build(structure,coefs,cubename,refcube,comm,size,rank):

    inp = ParseConfig().parse_input()

    (saltedname, saltedpath, saltedtype,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel, nspe1, nspe2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL) = ParseConfig().get_all_params()

    #if parallel:
    #    from mpi4py import MPI
    #    comm = MPI.COMM_WORLD
    #    size = comm.Get_size()
    #    rank = comm.Get_rank()
    #else:
    #    rank = 0
    #    size = 1

    # read basis
    [lmax,nmax] = basis.basiset(dfbasis)
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    lmax_max = max(llist)

    bohr2angs = 0.529177210670

    # read system
    ndata = len(structure)
    atomic_symbols = structure.get_chemical_symbols()
    valences = structure.get_atomic_numbers()
    coords = structure.get_positions()/bohr2angs
    cell = structure.get_cell()/bohr2angs

    # Define system excluding atoms that belong to species not listed in SALTED input 
    natoms_tot = len(atomic_symbols)
    excluded_species = []
    for iat in range(natoms_tot):
        spe = atomic_symbols[iat]
        if spe not in species:
            excluded_species.append(spe)
    excluded_species = set(excluded_species)
    for spe in excluded_species:
        atomic_symbols = list(filter(lambda a: a != spe, atomic_symbols))
        valences = list(filter(lambda a: a != spe, valences))
        coords = list(filter(lambda a: a != spe, coords))
    natoms = int(len(atomic_symbols))

    # get basis set info 
    bdir = osp.join(saltedpath,"basis")
    if rank==0: print("Reading auxiliary basis info...")
    alphas = {}
    sigmas = {}
    rcuts = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            avals = np.loadtxt(osp.join(bdir,f"{spe}-{dfbasis}-alphas-L{l}.dat"))
            if nmax[(spe,l)]==1:
                alphas[(spe,l,0)] = float(avals)
                sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                rcuts[(spe,l,0)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,0)]))*4
            else:
                for n in range(nmax[(spe,l)]):
                    alphas[(spe,l,n)] = avals[n]
                    sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
                    rcuts[(spe,l,n)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,n)]))*4

    naux = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe]+1):
             naux += nmax[(spe,l)]*(2*l+1)
    if rank==0: print("Number of auxiliary functions:", naux)
    
    # compute radial part of GTOs on a 1D mesh 
    ngrid = 20000
    interp_radial = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                dxx = rcuts[(spe,l,n)]/float(ngrid-1)
                rvec = np.zeros(ngrid)
                radial = np.zeros(ngrid)
                for irad in range(ngrid):
                    r = irad*dxx
                    rvec[irad] = r
                    radial[irad] = r**l * np.exp(-alphas[(spe,l,n)]*r**2)
                inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                radial /= np.sqrt(inner)
                interp_radial[(spe,l,n)] = interp1d(rvec,radial,kind="quadratic")
   
    if len(refcube)==1:

        # Read reference cube file
        cubefile = open(refcube[0] ,"r")
        lines = cubefile.readlines()
        nside = {}
        nside[0] = int(lines[3].split()[0])
        nside[1] = int(lines[4].split()[0])
        nside[2] = int(lines[5].split()[0])
        npoints = 1
        for i in range(3):
            npoints *= nside[i]
        if rank==0: print("Number of grid points:", npoints)
        dx = float(lines[3].split()[1])
        dy = float(lines[4].split()[2])
        dz = float(lines[5].split()[3])
        origin = np.asarray(lines[2].split(),dtype=float)[1:4]
        rho_qm = []
        for line in lines[6+natoms:]:
            rhovals = np.asarray(line.split(),float)
            for rhoval in rhovals:
                rho_qm.append(rhoval)
        cubefile.close()
        if npoints!=len(rho_qm):
           print("ERROR: inconsistent number of grid points!")
           sys.exit(0)
    
    elif len(refcube)==0:

        # Define cube parameters from scratch 
        nside = {}
        nside[0] = 200
        nside[1] = 200
        nside[2] = 200
        npoints = 1
        for i in range(3):
            npoints *= nside[i]
        dx = float(cell[0,0]/nside[0])
        dy = float(cell[1,1]/nside[1])
        dz = float(cell[2,2]/nside[2])
        origin = np.array([0.0,0.0,0.0])

    else:

        print("ERROR: please provide refcube as a single entry")
        sys.exit(0)
    
    if rank==0: print("Reconstructing the electron density in real space...")
   
    #Define 3D grid
    grid_regular=np.transpose(np.asarray(np.meshgrid(dx*np.asarray(range(nside[0])),
                                                     dy*np.asarray(range(nside[1])),
                                                     dz*np.asarray(range(nside[2])) ) ),(2,1,3,0))
    grid_regular=grid_regular.reshape((npoints,3))
   
    if parallel:

        if natoms < size:
            if rank == 0:
                raise ValueError(
                    f"More processes {size=} have been requested than atoms {natoms=}. "
                    f"Please reduce the number of processes."
                )
            else:
                exit()
        atoms_range = get_conf_range(rank, size, natoms, np.arange(natoms,dtype=int))
        atoms_range = comm.scatter(atoms_range, root=0)
        print(
            f"Task {rank+1} handles the following atoms: {atoms_range}", flush=True
        )

    else:

        atoms_range = np.arange(natoms,dtype=int) 

    natoms_range = int(len(atoms_range))
    coords_range = coords[atoms_range]
    atomic_symbols_range = [atomic_symbols[i] for i in atoms_range]

    ncoefs = 0
    for iat in range(natoms_range):
        spe = atomic_symbols_range[iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                ncoefs += 2*l+1

    coefs_range = np.zeros(ncoefs)
    iaux = 0
    itot = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if iat in atoms_range:
           iblock = 0
           for l in range(lmax[spe]+1):
               for n in range(nmax[(spe,l)]):
                   itemp = itot+iblock
                   coefs_range[iaux:iaux+2*l+1] = coefs[itemp:itemp+2*l+1]
                   iblock += 2*l+1
                   iaux += 2*l+1
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                itot += 2*l+1

    # compute 3D density
    rhor = np.zeros(npoints)
    iaux=0
    for iat in range(natoms_range):
        #print(iat+1,flush=True)
        spe = atomic_symbols_range[iat]
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                if rank==0: print("l=",l,"n=",n,flush=True)
                coef = coefs_range[iaux:iaux+2*l+1]
                repmax = np.zeros(3,int)
                for ix in range(3):
                    nreps = math.ceil(rcuts[(spe,l,n)]/cell[ix,ix])
                    if nreps < 1:
                        repmax[ix] = 1
                    else:
                        repmax[ix] = nreps
                if inp.qm.periodic=="3D":
                    for ix in range(-repmax[0],repmax[0]+1):
                        for iy in range(-repmax[1],repmax[1]+1):
                            for iz in range(-repmax[2],repmax[2]+1):
                                coord = coords_range[iat] - origin
                                coord[0] += ix*cell[0,0]
                                coord[1] += iy*cell[1,1]
                                coord[2] += iz*cell[2,2]
                                dvec = grid_regular - coord
                                d2list = np.sum(dvec**2,axis=1)
                                indx = np.where(d2list<=rcuts[(spe,l,n)]**2)[0]
                                nidx = len(indx)
                                rr = dvec[indx]
                                lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
                                lth = np.arccos(rr[:,2]/lr)
                                lph = np.arctan2(rr[:,1],rr[:,0])
                                #harmonics
                                ylm_real = np.zeros((2*l+1,nidx))
                                lm = 0
                                for m in range(-l,1):
                                    ylm = special.sph_harm(m,l,lph,lth)
                                    if m==0:
                                        ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                                        lm += l+1
                                    else:
                                        ylm_real[lm,:] = -np.imag(ylm)
                                        ylm_real[lm-2*m,:] = np.real(ylm)
                                        lm += 1
                                ylm_real *= np.sqrt(2.0)
                                #interpolate radial functions
                                radial_gto = interp_radial[(spe,l,n)](lr)
                                gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                                rhor[indx] += np.dot(coef,gto)
                elif inp.qm.periodic=="2D":
                    for ix in range(-repmax[0],repmax[0]+1):
                        for iy in range(-repmax[1],repmax[1]+1):
                            coord = coords_range[iat] - origin
                            coord[0] += ix*cell[0,0]
                            coord[1] += iy*cell[1,1]
                            dvec = grid_regular - coord
                            d2list = np.sum(dvec**2,axis=1)
                            indx = np.where(d2list<=rcuts[(spe,l,n)]**2)[0]
                            nidx = len(indx)
                            rr = dvec[indx]
                            lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
                            lth = np.arccos(rr[:,2]/lr)
                            lph = np.arctan2(rr[:,1],rr[:,0])
                            #harmonics
                            ylm_real = np.zeros((2*l+1,nidx))
                            lm = 0
                            for m in range(-l,1):
                                ylm = special.sph_harm(m,l,lph,lth)
                                if m==0:
                                    ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                                    lm += l+1
                                else:
                                    ylm_real[lm,:] = -np.imag(ylm)
                                    ylm_real[lm-2*m,:] = np.real(ylm)
                                    lm += 1
                            ylm_real *= np.sqrt(2.0)
                            #interpolate radial functions
                            radial_gto = interp_radial[(spe,l,n)](lr)
                            gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                            rhor[indx] += np.dot(coef,gto)
                elif inp.qm.periodic=="0D":
                    coord = coords_range[iat] - origin
                    dvec = grid_regular - coord
                    d2list = np.sum(dvec**2,axis=1)
                    indx = np.where(d2list<=rcuts[(spe,l,n)]**2)[0]
                    nidx = len(indx)
                    rr = dvec[indx]
                    lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
                    lth = np.arccos(rr[:,2]/lr)
                    lph = np.arctan2(rr[:,1],rr[:,0])
                    #harmonics
                    ylm_real = np.zeros((2*l+1,nidx))
                    lm = 0
                    for m in range(-l,1):
                        ylm = special.sph_harm(m,l,lph,lth)
                        if m==0:
                            ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                            lm += l+1
                        else:
                            ylm_real[lm,:] = -np.imag(ylm)
                            ylm_real[lm-2*m,:] = np.real(ylm)
                            lm += 1
                    ylm_real *= np.sqrt(2.0)
                    #interpolate radial functions
                    radial_gto = interp_radial[(spe,l,n)](lr)
                    gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                    rhor[indx] += np.dot(coef,gto)
                else:
                    print("ERROR: selected periodicity not implemented.")
                    sys.exit(0) 
                iaux += 2*l+1
  
    if parallel:
        # Reduce 3D field in slices to avoid MPI overflows 
        comm.Barrier()
        nslices = int(np.ceil(len(rhor) / 100.0))
        for islice in range(nslices-1):
            rhor[islice*100:(islice+1)*100] = comm.allreduce(rhor[islice*100:(islice+1)*100])
        rhor[(nslices-1)*100:] = comm.allreduce(rhor[(nslices-1)*100:])

    if rank==0:

        # compute integrated electronic charge
        nele = np.sum(rhor)*dx*dy*dz
        print("Integral = ", nele)

        # compute error as a fraction of electronic charge
        if refcube and saltedtype=="density":
            error = np.sum(abs(rhor-rho_qm))*dx*dy*dz/nele
            print("% MAE =", error*100)
        
        dirpath = os.path.join(saltedpath, "cubes")
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        
        # print density on a cube file
        cubef = open(saltedpath+"cubes/"+cubename,"w")
        print("Reconstructed electron density",file=cubef)
        print("CUBE FORMAT",file=cubef)
        print(natoms, origin[0], origin[1], origin[2],file=cubef)
        metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
        for ix in range(3):
            print(nside[ix], metric[ix,0], metric[ix,1], metric[ix,2],file=cubef)
        for iat in range(natoms):
            print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
        for igrid in range(npoints):
            print(rhor[igrid],file=cubef)
        cubef.close()

if __name__ == "__main__":
    build()

