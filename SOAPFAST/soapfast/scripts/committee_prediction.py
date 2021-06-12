#!/usr/bin/env python

import argparse
import sys,os
import numpy as np


def main():

    parser = argparse.ArgumentParser(description="Do spherical prediction with committee models")
    parser.add_argument("-w",  "--weights", type=str, required=True,            help="Weights file")
    parser.add_argument("-k",  "--kernel",  type=str, required=True,            help="Kernel file")
    parser.add_argument("-lm", "--lambda",  type=int, default=0,                help="Spherical order")
    parser.add_argument("-o",  "--ofile",   type=str, default="prediction.txt", help="Output file")
    args = parser.parse_args()

    # Read in kernel and weights
    ker = np.load(args.kernel)
    wts = np.load(args.weights)

    # Define important variables
    ofile = args.ofile
    lm    = args.lambda

    nmodel = len(wts)-2
    print("Doing prediction with a committee of ",str(nmodel)," models")
    alpha = wts[-1]
    mean  = wts[-2]
    wts   = np.array(wts[:nmodel]).astype(float)

    # Do prediction
    pred = np.dot(ker,wts)
    for i in range(len(wts[0])):
        pred[:,i] += mean[i]

    # Reshape prediction
    degen = 2*lm + 1
    pred = np.split(pred,len(ker))

    # Print prediction
    fl = open(ofile,'w')
    for i in range(len(ker)):
        print(' '.join(str(e) for e in predi]),file=fl)
    fl.close()

if __name__=="__main__":

    main()
