#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write
from apply_fps import apply_FPS

def generate_FPS(PS,scale = [],nsparse = 1000,ofile = '',initial=-1,verbose = False):

    # If we do not have a scalar power spectrum, reshape it
    if (not len(np.shape(PS)) in [3,4]):
        print("ERROR: power spectrum matrix has the wrong number of dimensions!")
        sys.exit(0)
    if (len(np.shape(PS)) == 4):
        PS = PS.reshape((len(PS),len(PS[0]),len(PS[0,0])*len(PS[0,0,0])))
    nrow = len(PS)

    # Have we given scaling factors?
    if (len(scale) == 0):
        # Scale everything by 1
        scale = [1 for i in range(len(PS))]
    
    # Go through and divide by the scaling factors
    PS_peratom = np.zeros((len(PS),len(PS[0,0])),dtype=float)
    for i in range(len(PS)):
        PS_peratom[i] = np.sum(PS[i],axis=0) / scale[i]
    
    # Do FPS sparsification
    iy = np.zeros(nsparse,int)

    # Choose initial row    
    if (initial == -1):
        iy[0] = np.random.randint(0,nrow)
    else:
        iy[0] = initial
    
    if verbose:
        print("Initial row: ",iy[0])
 
    # Get list of distances
    n2 = np.array([np.sum(PS_peratom[i] * np.conj([PS_peratom[i]])) for i in range(len(PS_peratom))])
    dl = n2 + n2[iy[0]] - 2*np.real(np.dot(PS_peratom,np.conj(PS_peratom[iy[0]])))
    for i in range(1,nsparse):
        if verbose:
            print("Doing ",i," of ",nsparse," dist = ",max(dl))
        # Choose the point that is furthest from all of the points we have already chosen
        iy[i] = np.argmax(dl)
        nd = n2 + n2[iy[i]] - 2*np.real(np.dot(PS_peratom,np.conj(PS_peratom[iy[i]])))
        dl = np.minimum(dl,nd)
    
    # Save FPS rows
    if ofile != '':
        np.save(ofile + "_rows.npy",iy)

    return iy

#################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Do FPS on power spectrum rows")
    parser.add_argument("-p", "--power",   type=str,  required=True,          help="Power spectrum file")
    parser.add_argument("-s", "--scaling", type=str,  default='',             help="Scaling file")
    parser.add_argument("-n", "--nrow",    type=int,  default=1000,           help="Number of rows to keep")
    parser.add_argument("-o", "--ofile",   type=str,  default="fps",          help="Output file")
    parser.add_argument("-i", "--initial", type=int,  default=-1,             help="Initial row")
    parser.add_argument("-v", "--verbose",            action='store_true',    help="Verbose mode")
    parser.add_argument("-a", "--apply",   type=str,  default='',             help="Apply FPS and print to this output file")
    args = parser.parse_args()
    
    # Firstly, check whether the number of rows to keep is larger than the number of rows in the power spectrum.
    PS = np.load(args.power)
    nrow = len(PS)
    nsparse = args.nrow
    
    if (args.scaling == ''):
        scale = np.array([1 for i in range(nrow)])
    else:
        scale = np.load(args.scaling)
    
    if (nsparse > nrow):
        print("ERROR: number of rows requested is larger than the number of rows in the power spectrum!")
        sys.exit(0)
    
    fps = generate_FPS(PS,scale=scale,nsparse=nsparse,ofile=args.ofile,initial=args.initial,verbose=args.verbose)

    if (args.apply != ''):
        print("Applying FPS")
        apply_fps.apply_FPS(PS,fps,ofile=args.apply)

if __name__=="__main__":

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from utils import regression_utils
    from apply_fps import apply_FPS

    main()
