#!/usr/bin/env python

import sys
import time
import ase
from ase.io import read
from ase.data import atomic_numbers,chemical_symbols
import numpy as np
import scipy
from scipy import special
from sympy.physics.wigner import wigner_3j
import random

###############################################################################################################################

def get_power_spectrum(lam,frames,nmax=8,lmax=6,rc=4.0,sg=0.3,ncut=-1,cw=1.0,periodic=False,outfile='',subset=['NO',None],initial=-1,sparsefile='',sparse_options=[''],cen=[],spec=[],atomic=[False,None],all_radial=[1.0,0.0,0.0],useall=False,xyz_slice=[],verbose=True,get_imag=False,norm=True,electro=False,sigewald=1.0,single_radial=False,radsize=50,lebsize=146):

    # If we want a slice of the coordinates, do this BEFORE anything else.
    if (len(xyz_slice)!=0):
        frames = frames[xyz_slice[0]:xyz_slice[1]]

    # Have we asked for a subset of the coordinates to be used?
    if (subset[0] == 'SEQ'):
        frames = frames[:subset[1]]
    elif (subset[0] == 'RANDOM'):
        random.shuffle(frames)
        frames = frames[:subset[1]]
    
    # Get coordinates and names
    npoints   = len(frames)
    all_names = np.array([frames[i].get_chemical_symbols() for i in range(npoints)])
    natmax    = len(max(all_names, key=len))
    coords    = [frames[i].get_positions() for i in range(npoints)]

    # Periodic or aperiodic?
    fixcell=False
    if periodic == True:
        cell = [frames[i].get_cell() for i in range(npoints)]
        for i in range(npoints):
            if np.allclose(cell[i],cell[0])==False:
                break
        if i+1==npoints:
            fixcell=True
            print("The cell is constant across configurations.")
    else:
        # Do a test here to make sure it really isn't periodic (and warn the user if it seems to be).
        if (np.linalg.det(frames[0].get_cell())!=0.0):
            print("\033[91m" + "WARNING: this input file has cell data; please check that it should be non-periodic!" + "\033[0m")
        cell = [np.array([0.0]) for i in range(npoints)]
    
    # List all species according to their valence
    all_species = []
    for k in list(set(np.concatenate(all_names))):
        all_species.append(atomic_numbers[k])
    
    # List number of atoms for each configuration
    nat = np.zeros(npoints,dtype=int)
    for i in range(npoints):
        nat[i] = len(all_names[i])

    # Get maximum number of neighbours according to frames density
    nnmax = 0
    if periodic==False:
        # Maximum number of atoms in one cluster (Safest option)
        nnmax = np.max([len(d) for d in frames]) 
    else: 
        max_density = max([len(d)*1.0/d.get_volume() for d in frames])
        nnmax = int(np.floor((4.0/3. *np.pi* rc**3) * max_density * 2))
#        nnmax = 1000

    if (verbose):
        print("Maximum number of neighbours = %i"%nnmax)
       
    # List indices for atom of the same species for each configuration
    all_indexes = []
    nsmax = len(atomic_numbers)
    for i in range(npoints):
        indexes = np.zeros((nsmax,natmax),int)
        for ispe in range(nsmax):
            idx = list(np.where(np.array(all_names[i][:nat[i]]) == chemical_symbols[ispe])[0])
            indexes[ispe,:len(idx)] = idx
        all_indexes.append(indexes)
    all_indexes = np.array(all_indexes)

    # Maximum number of nearest neighbours
    nneighmax = [np.array([len(list(np.where(np.array(all_names[i][:nat[i]]) == chemical_symbols[ispe])[0])) for ispe in range(nsmax)]) for i in range(npoints)]
   
    # Get list of centres 
    if len(cen) == 0 :
        centers = np.array(all_species)
    else:
        centers = np.array([atomic_numbers[i] for i in cen])
    
    if (verbose):
        print("selected centres: ", centers)

    # List number of atoms of chosen centres
    ncen = np.zeros(npoints,dtype=int)
    for i in range(npoints):
        ncen[i] = sum([[atomic_numbers[j] for j in all_names[i]].count(cens) for cens in centers])

    # Get list of species
    if (spec != []):
        all_species = list([atomic_numbers[spec[i]] for i in range(len(spec))])
    all_species = np.array(all_species)
    nspecies = len(all_species)

    if (verbose):
        print("selected species: ", all_species)
    
    # Setup orthogonal matrix
    [orthomatrix,sigma] = psutil.setup_orthomatrix(nmax,rc)
   
    # Setup arrays needed to save effort in tensorial power spectrum building 
    llmax = 0
    lvalues = {}
    if lam > 0:
        llmax=0
        lvalues = {}
        for l1 in range(lmax+1):
            for l2 in range(lmax+1):
                if (not get_imag):
                    # even combinations are considered
                    if (lam+l1+l2)%2==0 :
                        if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                            lvalues[llmax] = [l1,l2]
                            llmax+=1
                else:
                    # odd combinations are considered    
                    if (lam+l1+l2)%2!=0 :
                        if abs(l2-lam) <= l1 and l1 <= (l2+lam) :
                            lvalues[llmax] = [l1,l2] 
                            llmax+=1

    start = time.time()

    # Compute power spectrum of order lambda
    import os
    pname = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(pname + "/utils/LODE/gvectors.so"):
        [power,featsize] = psutil.compute_power_spectrum(nat,nneighmax,natmax,lam,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,all_indexes,all_species,coords,cell,rc,cw,sigma,sg,orthomatrix,sparse_options,all_radial,ncen,useall,verbose,electro,fixcell,sigewald,single_radial,radsize,lebsize)
    else:
        [power,featsize] = psutil.compute_power_spectrum(nat,nneighmax,natmax,lam,lmax,npoints,nspecies,nnmax,nmax,llmax,lvalues,centers,all_indexes,all_species,coords,cell,rc,cw,sigma,sg,orthomatrix,sparse_options,all_radial,ncen,useall,verbose)
    if (verbose):
        print("Power spectrum computed", time.time()-start) 
    
        print("starting feature space dimension", featsize)
        print("sparsifying power spectrum ....")
    start_sparse = time.time()
    
    # Sparsification options.
    if sparsefile != '':
        if (verbose):
            print("We have already sparsified the power spectrum with pre-loaded parameters.")
        psparse = power.reshape((2*lam+1)*npoints*natmax,len(sparse_options[1]))
        featsize = len(psparse[0])
    elif ncut > 0:
        # We have not provided a file with sparsification details, but we do want to sparsify, so we'll do it from scratch.
        if (verbose):
            print("Doing farthest-point sampling sparsification.")
        sparsefilename = "PS" + str(lam)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lmax"+str(lmax)+"_nmax"+str(nmax)+"_cutoff"+str(rc)+"_cweight"+str(cw)+"_init"+str(featsize)
        [psparse,sparse_details] = psutil.FPS_sparsify(power.reshape((2*lam+1)*npoints*natmax,featsize),featsize,ncut,initial)
        featsize = len(psparse[0])
        if (verbose):
            print("Saving sparsification details")
        sparsefilename += "_final"+str(featsize)
        if outfile != '':
            sparsefilename = outfile
        np.save(sparsefilename + "_fps.npy",sparse_details[0])
        np.save(sparsefilename + "_Amat.npy",sparse_details[1])
    else:
        if (verbose):
            print("Power spectrum will not be sparsified.")
        psparse = power.reshape((2*lam+1)*npoints*natmax,featsize)
    
    if (verbose):   
        print("done", time.time()-start_sparse) 
        print("final feature space dimension", featsize)

    # Make the power spectrum real and normalize it
    if (lam==0):
        power = psparse.reshape(npoints,natmax,featsize)
        power = np.real(power)
        if (norm):
            for i in range(npoints):
                for iat in range(ncen[i]):
                    inner = np.dot(power[i,iat],power[i,iat])
                    power[i,iat] /= np.sqrt(inner)
    else:
        power = psparse.reshape(npoints,natmax,2*lam+1,featsize)
        CC = np.conj(regression_utils.complex_to_real_transformation([2*lam+1])[0])
        if (not get_imag):
            # even combinations for l1+l2+lam are considered 
            power = np.real(np.einsum('ab,cdbe->cdae',CC,power))
        else:
            # odd combinations for l1+l2+lam are considered 
            power = np.imag(np.einsum('ab,cdbe->cdae',CC,power))
        if (norm):
            for i in range(npoints):
                for iat in range(ncen[i]):
                    inner = np.zeros((2*lam+1,2*lam+1),complex)
                    for mu in range(2*lam+1):
                        for nu in range(2*lam+1):
                            inner[mu,nu] = np.dot(power[i,iat,mu],np.conj(power[i,iat,nu]))
                    power[i,iat] /= np.sqrt(np.real(np.linalg.norm(inner)))


    # Reorder the power spectrum so that the ordering of the atoms matches their positions in the frame.
    for i in range(npoints):
        if (lam==0):
            ps_row = np.zeros((len(power[0]),len(power[0,0])),dtype=float)
        else:
            ps_row = np.zeros((len(power[0]),len(power[0,0]),len(power[0,0,0])),dtype=float)
        # Reorder this row
        reorder_list = [[] for cen in centers]
        atnum = frames[i].get_atomic_numbers()
        for j in range(len(atnum)):
            atom = atnum[j]
            if (atom in centers):
                place = list(centers).index(atom)
                reorder_list[place].append(j)
        reorder_list = sum(reorder_list,[])
        # The reordering list tells us where each column of the current power spectrum should go.
        for j in range(len(reorder_list)):
            ps_row[reorder_list[j]] = power[i,j]
        # Insert the reordered row back into the power spectrum
        power[i] = ps_row

    # Print power spectrum, if we have not asked for only a sample to be taken (we assume that taking a subset is just for the purpose of generating a sparsification)
    if (subset[0] == 'NO'):
        if (verbose):
            print("Saving power spectrum")
        PS_file = "PS"+str(lam)+"_nconf"+str(npoints)+"_sigma"+str(sg)+"_lmax"+str(lmax)+"_nmax"+str(nmax)+"_cutoff"+str(rc)+"_cweight"+str(cw)+"_ncut"+str(ncut)
        if (outfile != ''):
            PS_file = outfile
        # If we have asked for an atomic power spectrum, print this (or these).
        if (atomic[0] == True):
            tot_atoms = sum([len(all_names[i]) for i in range(npoints)])
            tot_atoms = sum(nat)
            if (lam==0):
                p_new = np.zeros((tot_atoms,1,len(power[0,0])),dtype=float)
            else:
                p_new = np.zeros((tot_atoms,1,2*lam+1,len(power[0,0,0])),dtype=float)
            k = 0
            for i in range(npoints):
                for j in range(nat[i]):
                    p_new[k,0] = power[i,j]
                    k += 1
            if (len(atomic[1])==0):
                # Print out all atoms that fall in the centers selection
                for atom in centers:
                    sym = chemical_symbols[atom]
                    atoms_list = np.where(np.concatenate([fr.numbers for fr in frames]) == atomic_numbers[sym])[0]
                    np.save(PS_file + "_atomic_" + sym + ".npy",p_new[atoms_list])
            else:
                # Just print out the selected atoms
                for atom in atomic[1]:
                    if (not atomic_numbers[atom] in centers):
                        print("Atom " + atom + " is not in list of centres!")
                    else:
                        atoms_list = np.where(np.concatenate([fr.numbers for fr in frames]) == atomic_numbers[atom])[0]
                        np.save(PS_file + "_atomic_" + atom + ".npy",p_new[atoms_list])
        # Otherwise, just print the total power spectrum
        else:
            np.save(PS_file + ".npy",power)

        # Print number of atoms, to be used with kernel building
        np.save(PS_file + '_natoms.npy',nat)
    
    if (verbose):
        print("Full calculation of power spectrum complete")

    return power

###############################################################################################################################

def main():

    # This is a wrapper that calls python scripts to build lambda-SOAP power spectra for use by SA-GPR.
    
    # Parse input arguments
    pname = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(pname + "/utils/LODE/gvectors.so"):
        args = parse.add_command_line_arguments_PS("Calculate power spectrum")
        [nmax,lmax,rcut,sig,cen,spec,cweight,lam,periodic,ncut,sparsefile,frames,subset,sparse_options,outfile,initial,atomic,all_radial,useall,xyz_slice,get_imag,nonorm,electro,sigewald,single_radial,radsize,lebsize] = parse.set_variable_values_PS(args)
        get_power_spectrum(lam,frames,nmax=nmax,lmax=lmax,rc=rcut,sg=sig,ncut=ncut,periodic=periodic,outfile=outfile,cw=cweight,subset=subset,initial=initial,sparsefile=sparsefile,sparse_options=sparse_options,cen=cen,spec=spec,atomic=atomic,all_radial=all_radial,useall=useall,xyz_slice=xyz_slice,get_imag=get_imag,norm=(not nonorm),electro=electro,sigewald=sigewald,single_radial=single_radial,radsize=radsize,lebsize=lebsize)
    else:
        args = parse.add_command_line_arguments_PS("Calculate power spectrum")
        [nmax,lmax,rcut,sig,cen,spec,cweight,lam,periodic,ncut,sparsefile,frames,subset,sparse_options,outfile,initial,atomic,all_radial,useall,xyz_slice,get_imag,nonorm] = parse.set_variable_values_PS(args)
        get_power_spectrum(lam,frames,nmax=nmax,lmax=lmax,rc=rcut,sg=sig,ncut=ncut,periodic=periodic,outfile=outfile,cw=cweight,subset=subset,initial=initial,sparsefile=sparsefile,sparse_options=sparse_options,cen=cen,spec=spec,atomic=atomic,all_radial=all_radial,useall=useall,xyz_slice=xyz_slice,get_imag=get_imag,norm=(not nonorm))

if __name__=="__main__":
    from utils import regression_utils
    import os
    pname = os.path.dirname(os.path.realpath(__file__))
    if os.path.isfile(pname + "/utils/LODE/gvectors.so"):
        import utils.LODE.PS_utils as psutil
        import utils.LODE.parsing as parse
    else:
        import utils.PS_utils as psutil
        import utils.parsing as parse
    main()
