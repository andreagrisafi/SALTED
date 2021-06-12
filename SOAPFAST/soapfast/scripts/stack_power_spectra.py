#!/usr/bin/env python

import os
import numpy as np

# Get environment variables
outfile = os.environ.get("outfile")
nrun = int(os.environ.get("numrun"))

# Read in power spectrum and number of atoms
all_ps=[np.load("PS_output_" + str(i+1) + ".npy") for i in range(nrun)]
all_natom=[np.load("PS_output_" + str(i+1) + "_natoms.npy") for i in range(nrun)]

# Put power spectra together
npoints = sum([np.shape(all_ps[i])[0] for i in range(nrun)])
natmax  = max([np.shape(all_ps[i])[1] for i in range(nrun)])
nfeat   = np.shape(all_ps[0])[-1]
if (len(np.shape(all_ps[0]))==3):
    degen=1
    intermediate_ps = [np.zeros((len(all_ps[i]),natmax,nfeat),float) for i in range(nrun)]
else:
    degen=len(all_ps[0][0,0])
    intermediate_ps = [np.zeros((len(all_ps[i]),natmax,degen,nfeat),float) for i in range(nrun)]

for i in range(nrun):
    intermediate_ps[i][:,:len(all_ps[i][0])] = all_ps[i]

# Free up memory used by all_ps
all_ps = None

new_ps = np.vstack(intermediate_ps)

# Print out the output
np.save(outfile + ".npy",new_ps)
np.save(outfile + "_natoms.npy",np.concatenate(all_natom))
