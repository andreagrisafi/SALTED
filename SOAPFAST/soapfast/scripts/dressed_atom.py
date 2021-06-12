#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
import scipy.optimize
from ase.io import read,write
import copy

def dressed_atom(features,property_in,ntrain,reg=-100.0,verbose=False):

    # Get training and testing features and properties
    train_feat = features[:ntrain]
    test_feat  = features[ntrain:]

    train_prop = property_in[:ntrain]
    test_prop  = property_in[ntrain:]

    # Build covariance
    covar = np.dot(train_feat.T,train_feat)

    def prediction(reg,prop):
        ereg = np.exp(reg)
        inv = np.dot(np.linalg.inv(covar + ereg*np.eye(len(covar))),train_feat.T)
        return np.dot(inv,prop)

    def get_error(reg,data,training_prop,testing_feat,iterate):
        wt = prediction(reg,training_prop)
        pred_data = np.dot(testing_feat,wt)
        test_data = data
        err = np.linalg.norm(pred_data - test_data) / len(pred_data)
        if (verbose):
            print("get_error called with reg = %f; error = %f"%(reg,err))
        if (iterate):
            return err
        else:
            return [err,wt]

    def func(x):
        return get_error(x,test_prop,train_prop,test_feat,True)

    # Minimize the prediction error with respect to the regularization
    outp = scipy.optimize.minimize(func,reg,method='Nelder-Mead',options={'initial_simplex':np.reshape([-10.0,1.0],(2,1))})
    print()
    print("Nelder-Mead optimization completed")
    out_reg = outp['x'][0]

    weights = get_error(out_reg,test_prop,train_prop,test_feat,False)[1]

    print(weights)

    # Return optimal weights and regularization
    return [weights,out_reg]

#########################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Dressed Atom Model",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--files",       required=True,                              help="Files for dressed-atom model")
    parser.add_argument("-p", "--property",    required=True,                              help="Property to model")
    parser.add_argument("-s", "--species",     required=False, default=[], nargs='+',      help="Species to use for model")
    parser.add_argument("-n", "--ntrain",      required=False, default=-1,                 help="Number of training points")
    parser.add_argument("-o", "--outfile",     required=False, default='',                 help="Output file name")
    parser.add_argument("-r", "--reg",         required=False, type=float, default=-100.0, help="Starting regularization")
    args = parser.parse_args()

    fname   = args.files
    prop    = args.property
    species = args.species
    ntrain  = args.ntrain
    outfile = args.outfile
    reg     = args.reg
    if (outfile == ''):
        outfile = fname
    outfile = outfile.replace(".xyz","")

    print("Dressed-atom model: converting property %s in file %s"%(prop,fname))

    # Read in file
    file_in = read(fname,':')
    npoints = len(file_in)

    # Get properties, take the trace if it's a rank-2 tensor
    tens = [file_in[i].info[prop] for i in range(npoints)]
    lnshp = len(np.shape(tens[0]))
    if (lnshp == 0):
        print("Scalar property")
        scalar = np.array([tens[i] for i in range(npoints)]).astype(float)
    elif (lnshp == 2):
        print("Rank-2 tensor property; taking trace")
        scalar = np.array([ (tens[i][0,0] + tens[i][1,1] + tens[i][2,2])/3.0 for i in range(npoints)]).astype(float)
    else:
        print("ERROR: only scalar or rank-2 tensor properties supported")
        sys.exit(0)

    # Get species
    if (len(species)==0):
        all_species = list(set(np.concatenate(np.array([file_in[i].get_chemical_symbols() for i in range(npoints)]))))
    else:
        all_species = species
    nspec = len(all_species)

    print("List of species:")
    print(all_species)

    # Get features
    ftrs = np.zeros((npoints,nspec),dtype=int)
    for i in range(npoints):
        ftrs[i] = np.array([file_in[i].get_chemical_symbols().count(spec) for spec in all_species])

    if (ntrain == -1):
        ntrain = int(len(file_in) * 0.75)
    else:
        ntrain = int(ntrain)

    # Call dressed-atom model
    model = dressed_atom(ftrs,scalar,ntrain,reg=reg,verbose=True)

    # Print output files
    np.save(outfile + "_dressed_atom.npy",np.array(model).astype(object))
    wt = model[0]
    avg = 0.0
    for i in range(npoints):
        dressed = np.dot(ftrs[i],wt)
        if (lnshp == 0):
            file_in[i].info[prop + '_DA'] = file_in[i].info[prop] - dressed
            avg += file_in[i].info[prop] - dressed
        elif (lnshp == 2):
            tensor = copy.deepcopy(file_in[i].info[prop])
            for j in range(3):
                tensor[j,j] -= dressed
            file_in[i].info[prop + '_DA'] = tensor
        else:
            print("ERROR: we should not be here!")
            sys.exit(0)
    write(outfile + "_dressed_atom.xyz",file_in)
    avg /= npoints
    print("Average is " + str(avg))

if __name__=="__main__":

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from utils import regression_utils

    main()
