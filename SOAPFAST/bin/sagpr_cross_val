#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write
from .do_fps import generate_FPS
from .apply_fps import apply_FPS
import random

def get_CV_set(frames,all_kernels,ncv,combine,dirroot=''):

    # Generate cross-validation sets. Firstly, randomly permute the frames
    ntot = len(frames)
    all_set = list(range(ntot))
    random.shuffle(all_set)
    np.save(dirroot + "CV_set.npy",all_set)

    # Split set into ncv parts
    CV_set = np.array_split(all_set,ncv)

    for i in range(ncv):
        print("Getting set ",i+1)
        # For each term in the sum, make this part into the training set and the remainder into the testing set
        cv_tr = []
        cv_te = CV_set[i]
        for j in range(ncv):
            if (j != i):
                cv_tr.append(CV_set[j])
        cv_tr = np.concatenate(cv_tr)
        # For this set, create training and testing coordinates
        dirname = dirroot + 'CV_' + str(i+1)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if os.path.exists(dirname + '/train.xyz'):
            os.remove(dirname + '/train.xyz')
        if os.path.exists(dirname + '/test.xyz'):
            os.remove(dirname + '/test.xyz')
        if os.path.exists(dirname + '/frames.xyz'):
            os.remove(dirname + '/frames.xyz')
        if (combine):
            tr_name = '/frames.xyz'
            te_name = '/frames.xyz'
        else:
            tr_name = '/train.xyz'
            te_name = '/test.xyz'
        for i in cv_tr:
            write(dirname + tr_name,frames[i],append=True)
        for i in cv_te:
            write(dirname + te_name,frames[i],append=True)
        # Split up the kernels
        if (all_kernels[0]=='K'):
            # Regular kernels
            for ker in all_kernels[1]:
                if (combine):
                    k_all = np.array([[ker[1][i,j] for i in np.concatenate([cv_tr,cv_te])] for j in np.concatenate([cv_tr,cv_te])]).astype(float)
                    np.save(dirname + '/' + ker[0].replace('.npy','_set.npy'),k_all)
                else:
                    ktr = np.array([[ker[1][i,j] for i in cv_tr] for j in cv_tr]).astype(float)
                    kte = np.array([[ker[1][i,j] for i in cv_tr] for j in cv_te]).astype(float)
                    np.save(dirname + '/' + ker[0].replace('.npy','_train.npy'),ktr)
                    np.save(dirname + '/' + ker[0].replace('.npy','_test.npy'),kte)
        else:
            # Sparsification kernels
            keep = True
            for ker in all_kernels[1]:
                if keep:
                    # This is a kernel between training set and environments
                    if (combine):
                        k_all = np.array([[ker[1][i,j] for i in np.concatenate([cv_tr,cv_te])] for j in range(len(ker[1][0]))]).astype(float)
                        np.save(dirname + '/' + ker[0].replace('.npy','_set.npy'),k_all)
                    else:
                        ktr = np.array([[ker[1][i,j] for i in cv_tr] for j in range(len(ker[1][0]))]).astype(float)
                        kte = np.array([[ker[1][i,j] for i in cv_te] for j in range(len(ker[1][0]))]).astype(float)
                        np.save(dirname + '/' + ker[0].replace('.npy','_train.npy'),ktr)
                        np.save(dirname + '/' + ker[0].replace('.npy','_test.npy'),kte)
                    keep = False
                else:
                    # This is a kernel between environments (just copy it as-is)
                    if (combine):
                        kname = 'set'
                    else:
                        kname = 'train'
                    np.save(dirname + '/' + ker[0].replace('.npy','_' + kname + '.npy'),ker[1])
                    keep = True

###########################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Generate cross-validation sets")
    parser.add_argument("-f",  "--frames",             required=True,              help="Atomic coordinates")
    parser.add_argument("-cv", "--crossval", type=int, required=True,              help="Number of cross-validation sets")
    parser.add_argument("-k",  "--kernel",             required=False, nargs='+',  help="Kernel files")
    parser.add_argument("-sf", "--sparse",             required=False, nargs='+',  help="Are we doing environmental sparsification?")
    parser.add_argument("-c",  "--combine",            action='store_true',        help="Combine training and test sets into one in each subfolder?")
    parser.add_argument("-d",  "--dirname",            required=False, default='', help="Directory root name")
    args = parser.parse_args()

    fname   = args.frames
    cval    = args.crossval
    kern    = args.kernel
    sparse  = args.sparse
    combine = args.combine
    dirroot = args.dirname

    if (kern == None and sparse == None):
        print("Either regular kernels or sparsification kernels must be specified!")
        sys.exit(0)
    if (kern != None and sparse != None):
        print("Either regular kernels or sparsification kernels must be specified (not both)!")
        sys.exit(0)

    frames = read(fname,':')
    all_kernels = ['',[]]
    if (kern != None):
        all_kernels[0] = 'K'
        for i in range(len(kern)):
            all_kernels[1].append([kern[i],np.load(kern[i])])
    else:
        all_kernels[0] = 'S'
        for i in range(len(sparse)):
            all_kernels[1].append([sparse[i],np.load(sparse[i])])

    get_CV_set(frames,all_kernels,cval,combine,dirroot=dirroot)

if __name__=="__main__":
    main()
