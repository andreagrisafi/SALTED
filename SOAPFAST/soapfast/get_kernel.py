#!/usr/bin/env python

import time
import numpy as np
import itertools
from utils import parsing,build_kernel,regression_utils

def get_kernel(PS,scale=[],PS0=[None,None],zeta=1,output='',use_hermiticity=False,verbose=True):

    start = time.time()
    
    # Get number of points in the power spectra
    npoints = [len(PS[0]),len(PS[1])]
    # Find degeneracy and lambda value
    if (len(np.shape(PS[0])) == 3):
        degen = 1
        lam = 0
    else:
        degen = len(PS[0][0,0])
        lam = int((degen-1) / 2)

    # If no scaling is given, replace this with arrays of 1s
    if (scale==[]):
        for j in range(2):
            scale.append(np.array([1 for i in range(npoints[j])]))

    if lam == 0:

        # Build scalar kernel

        featsize = [len(PS[0][0,0]),len(PS[1][0,0])]
        if (featsize[0] != featsize[1]):
            print("ERROR: number of features must be the same for the two power spectra!")
            sys.exit(0)
        featsize = featsize[0]

        if (verbose):
            print("Calculating scalar kernel with zeta=" + str(zeta))

        if zeta == 1: 

            # compute power spectrum average
            poweravg = [np.einsum('ab,a->ab',np.sum(PS[i],axis=1),1.0/scale[i]) for i in range(2)]
            # get kernel    
            kreal = np.real(np.dot(poweravg[0],np.conj(poweravg[1].T)))

        elif zeta > 1:

            # get kernel
            kreal = build_kernel.calc_nonlinear_scalar_kernel(npoints,scale,featsize,zeta,PS,use_hermiticity)

    elif lam > 0 :

        if (verbose):
            print("Calculating lambda=" + str(lam) + " kernel with zeta=" + str(zeta))

        # Build tensor kernel

        featsize = [len(PS[0][0,0,0]),len(PS[1][0,0,0])]
        if (featsize[0] != featsize[1]):
            print("ERROR: number of features must be the same for the two power spectra!")
            sys.exit(0)
        featsize = featsize[0]

        # Normalize by number of atoms
        PS = [np.einsum('cdbe,c->cdbe',PS[i],1.0/scale[i]) for i in range(2)]

        if zeta == 1: 

            # compute power spectrum average
            poweravg = [np.sum(PS[i],axis=1) for i in range(2)]
            # get kernel
            kreal    = build_kernel.calc_linear_tensor_kernel(lam,npoints,featsize,poweravg,use_hermiticity)

        elif zeta > 1:

            # get kernel
            featsize0 = len(PS0[0][0,0])
            kreal     = build_kernel.calc_nonlinear_tensor_kernel(lam,npoints,scale,featsize,featsize0,zeta,PS,PS0,use_hermiticity)

    if use_hermiticity:
        # Build the lower-triangular part of the kernel using the fact that it is hermitian.
        for i in range(npoints[0]):
            for j in range(i):
                kreal[i,j] = kreal[j,i].T

    # Save kernel 
    if output != '':
        np.save(output + str(".npy"),kreal)

    if (verbose):    
        print("Kernel computed", time.time() - start, "seconds")

    return kreal

###############################################################################################################################

def main():

    # This is a wrapper that calls python scripts to build lambda-SOAP kernels for use by SA-GPR.
    
    args = parsing.add_command_line_arguments_kernel("Calculate kernel")
    [power,scalefac,power0,zt,use_hermiticity] = parsing.set_variable_values_kernel(args)
    
    get_kernel(power,scale=scalefac,PS0=power0,zeta=zt,output=args.output,use_hermiticity=use_hermiticity)

if __name__=="__main__":
    from utils import parsing,build_kernel,regression_utils
    main()
