#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write

############################################################################################################

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
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
        dl = np.minimum(dl,nd)
    return iy

############################################################################################################

def do_feature_fps(PS,ncut=1000,initial=-1):

    # Reshape power spectrum
    npoints = len(PS)
    natmax  = len(PS[0])
    if (len(np.shape(PS))==3):
        degen = 1
        featsize = len(PS[0,0])
    else:
        degen = len(PS[0,0])
        featsize = len(PS[0,0,0])

    PS = PS.reshape(degen * npoints * natmax,featsize)

    # Get FPS vector
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
    psparse  = np.array([PS.T[i] for i in vec_fps]).T
    psparse  = np.dot(psparse,A_matrix)
    featsize = len(psparse[0])

    # Reshape sparsified power spectrum
    if (degen == 1):
        psparse = psparse.reshape(npoints,natmax,featsize)
    else:
        psparse = psparse.reshape(npoints,natmax,degen,featsize)

    # Return the sparsification vector (which we will need for later sparsification) and the A matrix (which we will need for recombination).
    sparse_details = [vec_fps,A_matrix]

    return [psparse,sparse_details]

############################################################################################################

def main():

    parser = argparse.ArgumentParser(description="FPS sparsification on feature space")
    parser.add_argument("-p", "--power", type = str, required=True, help="Power spectrum file")
    parser.add_argument("-n", "--ncut", type=int, required=False, default=1000, help="Number of features")
    parser.add_argument("-i", "--initial", type=int, required=False, default=-1, help="Initial feature")
    args = parser.parse_args()

    # Read in the power spectrum
    PS = np.load(args.power + ".npy")
    ncut = args.ncut
    initial = args.initial

    # Do FPS
    [psparse,sparse_details] = do_feature_fps(PS,ncut=ncut,initial=initial)

    # Print out sparse details and new power spectrum
    np.save(args.power + "_sparse.npy",psparse)
    np.save(args.power + "_fps.npy",sparse_details[0])
    np.save(args.power + "_Amat.npy",sparse_details[1])

if __name__=="__main__":
    main()
