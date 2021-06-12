#!/usr/bin/env python

from ase.io import read,write
import argparse
import os
import math

parser = argparse.ArgumentParser(description="Separate xyz file into blocks for power spectrum calculation")
parser.add_argument("-f", "--file",   type=str,    required=True, help="Files to read in")
parser.add_argument("-n", "--nblock", type=int,    required=True, help="Number of blocks")
parser.add_argument("-o", "--ofile", type=str, default="slice", help="Output file prefix")
args = parser.parse_args()

nblock = args.nblock
infile = args.file

# Get coordinate file, number of configurations and size and number of blocks
all_coords = read(infile,':')
nconfigs = len(all_coords)
blocksize = int(math.ceil(float(nconfigs)/float(nblock)))
nblock = int(math.ceil(float(nconfigs)/float(blocksize)))

print("Read in file with %i frames."%(nconfigs))
print("Creating %i blocks."%(nblock))
print("Each block will contain (up to) %i frames."%(blocksize))

# Split into blocks
for i in range(nblock):
    fname = args.ofile + "_" + str(i) + ".xyz"
    imin = i*blocksize
    imax = min(i*blocksize + blocksize,nconfigs)
    coords_out = [all_coords[i] for i in range(imin,imax)]
    write(fname,coords_out)
