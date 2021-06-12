#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write

def cartesian_to_spherical(tens,ftrs,rank = -1,outfile = '', threshold = 1e-6, property_to_convert = ''):

    ndata = len(tens)
    num_elem = len(tens[0].split())

    # If we are not given a rank, infer it based on the dimension of the input tensor
    if rank == -1:
        rank = int(np.log(num_elem) / np.log(3.0))
    
    # Get Cartesian to spherical matrix as well as list of components that we are keeping
    [all_CS,keep_cols,all_sym] = regression_utils.get_cartesian_to_spherical(rank)
    
    CS = all_CS[-1]
    
    # Find degeneracies and list of how to split up the transformed spherical tensors
    degen = [1 + 2*keep_cols[-1][i][-1] for i in range(len(keep_cols[-1]))]
    cumulative_degen = [sum(degen[:i]) for i in range(1,len(degen)+1)]
    
    # Get complex to real transformations
    CR = regression_utils.complex_to_real_transformation(degen)
    
    all_tens_sphr = [ [] for i in range(len(degen))]
    for i in range(len(tens)):
        # Split into spherical components
        tens_sphr = np.split(np.dot(np.array(tens[i].split()).astype(float),CS),cumulative_degen)
        for j in range(len(degen)):
            spherical = tens_sphr[j]
            # Is this a regular or imaginary spherical tensor?
            if (not all_sym[-1][j]):
                # Convert to regular spherical tensor
                spherical /= 1.0j
            # Transform to real spherical harmonic
            out_tens = np.dot(CR[j],spherical)
            if (degen[j]==1):
                out_tens = out_tens[0]
            all_tens_sphr[j].append(out_tens)
    
    # Print out these components to files
    if outfile != '':
        for i in range(len(keep_cols[-1])):
            if (np.linalg.norm(all_tens_sphr[i][:]) > threshold):
                # If we have an L=1 spherical harmonic, then the descriptor string needs to be changed
                if (len(keep_cols[-1][i]) == 1):
                    keep_num = '1'
                else:
                    keep_num = ''.join(map(str,keep_cols[-1][i][1:]))
                prop_out = property_to_convert + "_L" + keep_num
                print("  Outputting property %s to %s"%(prop_out,outfile))
                for j in range(ndata):
                    to_print = np.real(all_tens_sphr[i][j])
                    ftrs[j].info[prop_out] = to_print
        
        write(outfile,ftrs)
    else:
        return ftrs

#########################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Convert Cartesian to spherical tensors",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--files",                 required=True, help="Files to convert")
    parser.add_argument("-o", "--output",                default='',    help="Output file")
    parser.add_argument("-p", "--property",              required=True, help="Property to convert")
    parser.add_argument("-r", "--rank",                  required=True, help="Tensor rank")
    parser.add_argument("-t", "--threshold", type=float, default=1e-6,  help="Threshold for tensor values being zero")
    args = parser.parse_args()
    
    file_to_convert = args.files
    property_to_convert = args.property
    rank = int(args.rank)
    outfile = args.output
    
    if outfile == '':
        outfile = file_to_convert
    
    print("Converting rank-%i property %s in file %s"%(rank,property_to_convert,file_to_convert))
    
    # Read in tensor file
    ftrs = read(file_to_convert,':')
    if rank == 0:
        tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
    elif rank == 2:
        tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
    else:
        tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
    
    cartesian_to_spherical(tens,ftrs,rank=rank,outfile=outfile,threshold=args.threshold,property_to_convert=property_to_convert)

if __name__=="__main__":

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from utils import regression_utils

    main()
