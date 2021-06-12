#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write
from do_fps import generate_FPS
from apply_fps import apply_FPS
import random

def get_training_set(PS,frames,scale = [],fps = False,initial = -1,ntrain = -1,mode = 'seq',infile = '',outfile = ''):

    FPS_details      = []
    reordered_PS     = PS
    reordered_frames = frames
    if fps:
        # Do FPS ordering of the power spectrum
        FPS_details      = generate_FPS(PS[0],scale=scale,nsparse=len(PS[0]),initial=initial)
        reordered_PS     = [apply_FPS(PS[i],FPS_details) for i in range(len(PS))]
        reordered_frames = [frames[i] for i in FPS_details]
    
    # Choose the training set
    if (ntrain > len(PS[0])) or (ntrain == -1):
        ntrain = len(PS[0])
    
    if   (mode == 'seq'):
        training_set = np.array(list(range(ntrain)))
    elif (mode == 'rdm'):
        training_set = list(range(len(PS[0])))
        random.shuffle(training_set)
        training_set = np.array(training_set[:ntrain])
    elif (mode == 'input'):
        training_set = np.array([int(line.rstrip()) for line in open(infile)])
    else:
        print("ERROR: I don't see how we got here.")
        sys.exit(0)
    
    # Get the testing set
    testing_set = np.setdiff1d(list(range(len(PS[0]))),training_set)
    
    # Split the power spectrum into testing and training power spectra
    PS_train    = [PS[i][training_set] for i in range(len(PS))]
    PS_test     = [PS[i][testing_set] for i in range(len(PS))]
    frame_train = [frames[i] for i in training_set]
    frame_test  = [frames[i] for i in testing_set]
    
    # Now print out all of the important information: FPS ordering (if applicable), training and testing sets, as well as training and testing power spectra
    out_array = [FPS_details,reordered_frames,training_set,testing_set,frame_train,frame_test]
    for i in range(len(PS)):
        out_array.append(PS_train[i])
        out_array.append(PS_test[i])
    out_array = np.array(out_array,dtype=object)
    if outfile != '':
        np.save(outfile + '_sets.npy',out_array)
    else:
        return out_array


###########################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Generate training and testing sets")
    parser.add_argument("-p",  "--power",             required=True, nargs='+',                     help="Power spectrum file")
    parser.add_argument("-fr", "--frames",            required=True,                                help="Atomic coordinates")
    parser.add_argument("-s",  "--scaling",           default=[],                                   help="Scaling file")
    parser.add_argument("-i",  "--initial", type=int, default=-1,                                   help="Initial row")
    parser.add_argument("-fp", "--fps",               action='store_true',                          help="Do FPS ordering of the power spectrum?")
    parser.add_argument("-m",  "--mode",              choices=['seq','rdm','input'], default="seq", help="Mode for choosing the training set")
    parser.add_argument("-n",  "--ntrain",  type=int, default=-1,                                   help="Number of training points")
    parser.add_argument("-f",  "--infile",            default='',                                   help="Input file for training set")
    parser.add_argument("-o",   "--ofile",            default='',                                   help="Output file prefix")
    args = parser.parse_args()
    
    PS      = [np.load(args.power[i]) for i in range(len(args.power))]
    fps     = args.fps
    mode    = args.mode
    infile  = args.infile
    initial = args.initial
    outfile = args.ofile
    ntrain  = args.ntrain
    if ((mode == 'input') and (infile == '')):
        print("ERROR: an input file must be specified!")
        sys.exit(0)
    
    if (outfile == ''):
        outfile = args.power[0].replace('.npy','')

    nrow = len(PS[0])
    
    if (args.scaling == ''):
        scale = np.array([1 for i in range(nrow)])
    else:
        scale = np.load(args.scaling)

    frames = read(args.frames,':')

    get_training_set(PS,frames,scale=scale,fps=fps,initial=initial,ntrain=ntrain,mode=mode,infile=infile,outfile=outfile)

if __name__=="__main__":
    main()
