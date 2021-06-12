#!/usr/bin/env python

import argparse
import numpy as np
import random,sys
from copy import deepcopy
from ase.io import read,write

parser = argparse.ArgumentParser(description="Subsample for uncertainty estimation")
parser.add_argument("-k",  "--kernel",   type=str, required=True, help="Kernel file")
parser.add_argument("-f",  "--fname",    type=str, required=True, help="File name")
parser.add_argument("-np", "--npoints",  type=int, required=True, help="Number of points per subsample")
parser.add_argument("-ns", "--nsamples", type=int, default=1,     help="Number of subsamples")
args = parser.parse_args()

kernel = np.load(args.kernel)
xyz = read(args.fname,':')

if (len(xyz)!=len(kernel)):
    print("ERROR: xyz file and kernel have different lengths!",len(kernel),len(xyz))
    sys.exit(0)

for i in range(args.nsamples):
    idx = [i for i in range(len(xyz))]
    random.shuffle(idx)
    xy2 = deepcopy(xyz)
    ke2 = deepcopy(kernel)
    for j in range(len(idx)):
        xy2[j] = xyz[idx[j]]
        ke2[j] = kernel[idx[j]]
    idx = idx[:args.npoints]
    xy2 = xy2[:args.npoints]
    ke2 = ke2[:args.npoints]
    np.save("IDX." + str(i+1) + ".npy",idx)
    np.save("KERNEL." + str(i+1) + ".npy",ke2)
    write("FRAMES." + str(i+1) + ".xyz",xy2)
