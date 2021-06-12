#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import utils.regression_utils
import utils.sagpr_utils
from ase.io import read,write
from sympy.physics.quantum.cg import CG

parser = argparse.ArgumentParser(description="List spherical tensor components",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--files",                 required=True, help="Files to convert")
parser.add_argument("-o", "--output",                default='',    help="Output file")
parser.add_argument("-p", "--property",              required=True, help="Property to convert")
parser.add_argument("-r", "--rank",                  required=True, help="Tensor rank")
parser.add_argument("-t", "--threshold", type=float, default=1e-6, help="Threshold for tensor values being zero")
args = parser.parse_args()

file_to_convert = args.files
property_to_convert = args.property
rank = int(args.rank)
outfile = args.output
threshold = args.threshold

# Read in tensor file
ftrs = read(file_to_convert,':')
if rank == 0:
    tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
elif rank == 2:
    tens = [' '.join(np.concatenate(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
else:
    tens = [' '.join(np.array(ftrs[i].info[args.property]).astype(str)) for i in range(len(ftrs))]
ndata = len(tens)
num_elem = len(tens[0].split())

# Get Cartesian to spherical transformation and a list of the components we are keeping
[all_CS,keep_cols,all_sym] = utils.regression_utils.get_cartesian_to_spherical(rank)

CS = all_CS[-1]

# Get list of degeneracies and array describing how to split up spherical tensor output
degen = [1 + 2*keep_cols[-1][i][-1] for i in range(len(keep_cols[-1]))]
cumulative_degen = [sum(degen[:i]) for i in range(1,len(degen)+1)]

CR = utils.regression_utils.complex_to_real_transformation(degen)

all_tens_sphr = [ [] for i in range(len(degen))]
for i in range(len(tens)):
    # Split into spherical components
    tens_sphr = np.split(np.dot(np.array(tens[i].split()).astype(float),CS),cumulative_degen)
    for j in range(len(degen)):
        spherical = tens_sphr[j]
        # Convert to real spherical harmonic
        out_tens = np.dot(CR[j],spherical)
        if (degen[j]==1):
            out_tens = out_tens[0]
        all_tens_sphr[j].append(out_tens)

for i in range(len(degen)):
    if (degen[i] > 1):
        all_tens_sphr[i] = np.reshape(all_tens_sphr[i],np.size(all_tens_sphr[i]))

# Decide which components to keep
spherical_components = []
keep_list            = []
out_degen            = []
out_CR               = []
keep_indices         = []
keep_sym             = []
for i in range(len(degen)):
    if (np.linalg.norm(all_tens_sphr[i][:]) > threshold):
        spherical_components.append(all_tens_sphr[i])
        keep_list.append(keep_cols[-1][i])
        out_degen.append(degen[i])
        out_CR.append(CR[i])
        keep_sym.append(all_sym[-1][i])
        keep_indices.append(i)

# Check to see if any of the components with the same degeneracy are linearly independent (as happens, e.g., for L=3 symmetric tensors)
lin_dep_list  = []
lin_dep_local = []
for i in range(len(keep_indices)):
    for j in range(i+1,len(keep_indices)):
        if (out_degen[i] == out_degen[j]):
            if (np.linalg.matrix_rank(np.column_stack((spherical_components[i],spherical_components[j]))) <= 1):
                avg_list = []
                for k in range(len(spherical_components[i])):
                    if (abs(spherical_components[i][k]) > threshold):
                        avg_list.append(np.real(spherical_components[j][k] / spherical_components[i][k]))
                scale_factor = np.mean(avg_list)
                lin_dep_list.append([keep_cols[-1].index(keep_list[i]),keep_cols[-1].index(keep_list[j]),scale_factor])
                lin_dep_local.append(j)

# If any of the components are linearly dependent on each other, we don't need to keep both
final_components = []
final_keep       = []
final_degen      = []
final_CR         = []
final_indices    = []
final_sym        = []
for i in range(len(out_degen)):
    if (not i in lin_dep_local):
        final_components.append(spherical_components[i])
        final_keep.append(keep_list[i])
        final_degen.append(out_degen[i])
        final_CR.append(out_CR[i])
        final_sym.append(keep_sym[i])
        final_indices.append(keep_indices[i])

for i in range(len(final_degen)):
    if (final_sym[i]):
        print("L=%i component (%s), real"%((final_degen[i]-1)/2,''.join(map(str,final_keep[i][1:]))))
    else:
        print("L=%i component (%s), imaginary"%((final_degen[i]-1)/2,''.join(map(str,final_keep[i][1:]))))

# If desired, print this information to a file
if (outfile != ''):
    np.save(outfile,np.array([final_components,final_keep,final_CR,final_degen,lin_dep_list,CS],dtype=object))
