#!/usr/bin/env python

import argparse
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from ase.io import read,write
from ase.data import atomic_numbers,chemical_symbols

###############################################################################################################################

def complex_to_real_transformation(sizes):
    # Transformation matrix from complex to real spherical harmonics

    matrices = []
    for i in range(len(sizes)):
        lval = (sizes[i]-1)/2
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range( (sizes[i]-1)/2 ):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[(sizes[i]-1)/2][(sizes[i]-1)/2] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

###############################################################################################################################

def get_atomic_power_spectrum(PS,xyz,lam = 0,ofile = '', cen = ''):

    # Given a molecular power spectrum and its respective coordinate file, get the power spectrum for the atom-centred environments
    natoms = [len(xyz[i].get_chemical_symbols()) for i in range(len(xyz))]
    tot_atoms = sum(natoms)

    # Do a backwards compatibility check: if we have a complex power spectrum, convert it to real
    if (PS.dtype == complex):
        if (lam == 0):
            PS = np.real(PS)
        else:
            CC = np.conj(complex_to_real_transformation([2*lam+1])[0])
            PS = np.real(np.einsum('ab,cdbe->cdae',CC,PS))

    if (lam == 0):
        # Scalar power spectrum
        p_new = np.zeros((tot_atoms,1,len(PS[0,0])),dtype=float)
        k = 0
        # Fill the members of the new power spectrum with those of the old
        for i in range(len(xyz)):
            for j in range(natoms[i]):
                p_new[k,0] = PS[i,j]
                k += 1
    else:
        # Tensor power spectrum
        p_new = np.zeros((tot_atoms,1,len(PS[0,0]),len(PS[0,0,0])),dtype=float)
        k = 0
        # Fill the members of the new power spectrum with those of the old
        for i in range(len(xyz)):
            for j in range(natoms[i]):
                p_new[k,0] = PS[i,j]
                k += 1
    
    # If we have specified centres, print out the requisite centres
    if (ofile != ''):
        if cen != '':
            for centre in cen:
                # Go through our atomic PS and get the rows that correspond to this element
                atoms_list = np.where(np.concatenate([fr.numbers for fr in xyz]) == atomic_numbers[centre])[0]
                ps_atomic = p_new[atoms_list]
                np.save(ofile + '_' + centre + '.npy',ps_atomic)
        else:
            # Save new power spectrum
            np.save(ofile + '.npy',p_new)
    else:
        if cen != '':
            atomic_output = []
            for centre in cen:
                # Go through our atomic PS and get the rows that correspond to this element
                atoms_list = np.where(np.concatenate([fr.numbers for fr in xyz]) == atomic_numbers[centre])[0]
                ps_atomic = p_new[atoms_list]
                atomic_output.append(ps_atomic)
            return atomic_output
        else:
            return p_new

#############################################################################################

def main():

    # INPUT ARGUMENTS.
    parser = argparse.ArgumentParser(description="Get atomic power spectrum")
    parser.add_argument("-p",  "--power",   required=True,                           help="Power spectrum")
    parser.add_argument("-f",  "--fname",   required=True,                           help="Coordinates")
    parser.add_argument("-o",  "--ofile",   required=True, default='atomic_PS',      help="Output file")
    parser.add_argument("-c",  "--centres", required=False, default='', nargs='+',   help="Centres to use")
    args = parser.parse_args()
    
    pspec = args.power
    fname = args.fname
    ofile = args.ofile
    cen   = args.centres

    # Read in power spectrum and coordinate file, get number of atoms and the total
    PS = np.load(pspec)
    xyz = read(fname,':')

    # Find lambda value
    if (not len(np.shape(PS)) in [3,4]):
        print("ERROR: power spectrum matrix has the wrong number of dimensions!")
        sys.exit(0)
    if (len(np.shape(PS)) == 3):
        lam = 0
    else:
        lam = (len(PS[0,0]) - 1) / 2
    
    get_atomic_power_spectrum(PS,xyz,lam=lam,ofile=ofile,cen=cen)

if __name__=="__main__":
    main()

