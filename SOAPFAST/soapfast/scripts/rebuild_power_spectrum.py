#!/usr/bin/env python
import numpy as np
import sys
import argparse
import math
import time
from ase.io import read

parser = argparse.ArgumentParser(description="Rebuild power spectrum from blocks.")
parser.add_argument("-lm",  "--lam", type=int, required=True,    help="Spherical tensor order.")
parser.add_argument("-c", "--coords", type=str, required=True, help="Coordinates file.")
parser.add_argument("-nb", "--nblocks",   type=int,   required=True, help="Number of blocks.")
parser.add_argument("-f", "--fname", type=str, default="slice",help="Input file prefix.")
args = parser.parse_args()

lam = int(args.lam)
coords = str(args.coords)
nblocks = int(args.nblocks)
fname = str(args.fname)

all_coords = read(coords,':')
ndata = len(all_coords)

blocksize = int(math.ceil(float(ndata)/float(nblocks)))

# Put power spectra of individual blocks together into a single power spectrum
pslices = []
for i in range(nblocks):
    pslices.append(np.load(fname + "_" + str(i) + ".npy"))

power = np.vstack(pslices)

np.save(fname + ".npy",power)
