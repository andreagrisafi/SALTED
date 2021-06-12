#!/usr/bin/env python

import argparse
import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import numpy as np
from ase.io import read,write

def apply_FPS(PS,fps,ofile = ''):

    # Is this a scalar or a tensor kernel?
    scalar = (len(np.shape(PS)) == 3)
    
    if (not(scalar)):
        ll = len(PS[0,0])

    # Do a backwards compatibility check: if we have a complex power spectrum, convert it to real
    if (PS.dtype == complex):
        if (scalar):
            PS = np.real(PS)
        else:
            CC = np.conj(regression_utils.complex_to_real_transformation([ll])[0])
            PS = np.real(np.einsum('ab,cdbe->cdae',CC,PS))
    
    # Create arrays for output.
    if scalar:
        PS_new = np.zeros((len(fps),len(PS[0]),len(PS[0,0])),dtype=float)
    else:
        PS_new = np.zeros((len(fps),len(PS[0]),ll,len(PS[0,0,0])),dtype=float)
        
    # Get sparse power spectrum.
    for i in range(len(fps)):
        PS_new[i] = PS[fps[i]]
    
    # Save power spectrum.
    if (ofile != ''):
        np.save(ofile + '.npy',PS_new)
    else:
        return PS_new

############################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Apply FPS to power spectrum rows")
    parser.add_argument("-p", "--power", required=True, help="Power spectrum file")
    parser.add_argument("-sf", "--sparsefile", default='fps_rows', help="Sparse details file")
    parser.add_argument("-o", "--ofile", type=str, default='PS_fps', help="Output file")
    args = parser.parse_args()
    
    # Read in the power spectrum and the FPS details.
    PS = np.load(args.power)
    fps = np.load(args.sparsefile + '.npy')
    apply_FPS(PS,fps,ofile=args.ofile)

if __name__=="__main__":
    main()
