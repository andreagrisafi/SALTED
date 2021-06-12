#!/usr/bin/env python

import argparse
from scipy import optimize
import random
import sys,os
import numpy as np
from ase.io import read,write
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import regression_utils,sagpr_utils

###############################################################################################################################

def main():

    # This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.
    
    # Parse input arguments
    parser = argparse.ArgumentParser(description="SA-GPR",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-r",   "--rank",                type=str,   required=True,                              help="Rank of tensor to learn")
    parser.add_argument("-reg",  "--regularization",     type=float, required=True,  nargs='+',                  help="Lambda values list for KRR calculation")
    parser.add_argument("-ftr", "--ftrain",              type=float, default=1.0,                                help="Fraction of data points used for testing")
    parser.add_argument("-f",   "--features",            type=str,   required=True,                              help="File containing atomic coordinates")
    parser.add_argument("-p",   "--property",            type=str,   required=True,                              help="Property to be learned")
    parser.add_argument("-k",   "--kernel",              type=str,   required=False,  nargs='+',                 help="Files containing kernels")
    parser.add_argument("-sel", "--select",              type=str,   default=[],     nargs='+',                  help="Select maximum training partition")
    parser.add_argument("-rdm", "--random",              type=int,   default=0,                                  help="Number of random training points")
    parser.add_argument("-w",    "--weights",            type=str,   default='weights',                          help="File prefix to print out weights")
    parser.add_argument("-perat","--peratom",                        action='store_true',                        help="Call for scaling the properties by the number of atoms")
    parser.add_argument("-pr",   "--prediction",                     action='store_true',                        help="Carry out prediction?")
    parser.add_argument("-sf",   "--sparsify",           type=str,   required=False,  nargs='+',                 help="Kernels for sparsification")
    parser.add_argument("-m",    "--mode",               type=str,   choices=['solve','pinv'], default='solve',  help="Mode to use for inversion of kernel matrices")
    parser.add_argument("-c",    "--center",             type=str,   default='',                                 help="Species to be used for property extraction ")
    parser.add_argument("-ftol", "--functol",            type=float, default=0.0001,                             help="Tolerance for Nelder-Mead minimization")
    parser.add_argument("-prec", "--precision",          type=int,   default=1,                                  help="Precision of output")
    args = parser.parse_args()

    ftr = args.ftrain 
    spherical = True
    # Get regularization
    reg = args.regularization
    if (len(reg) != 2):
        print("ERROR: a minimum and maximum regularization must be specified!")
        sys.exit(0)
    rank = args.rank
    int_rank = int(rank[-1])
    func_tol = args.functol
 
    # Read in features
    ftrs = read(args.features,':')

    # Either we have supplied kernels for carrying out the regression, or sparsification kernels, but not both (or neither).
    kernels = args.kernel
    sparsify = args.sparsify
    nat = []
    [nat.append(ftrs[i].get_number_of_atoms()) for i in range(len(ftrs))]

    # Read in tensor data for training the model
    if args.center != '':
        if int_rank == 0:
            tens = [ str(frame_prop) for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]] ]
        else:
            tens = [' '.join(frame_prop.astype(str).reshape(2*int_rank + 1))  for fr in ftrs for frame_prop in fr.arrays[args.property][np.where(fr.numbers==atomic_numbers[args.center])[0]]]
        nat = [1 for i in range(len(tens))]
    elif args.peratom:
        if int_rank == 0:
            tens = [str(ftrs[i].info[args.property]/nat[i]) for i in range(len(ftrs))]
        else:
            tens = [' '.join((np.array(ftrs[i].info[args.property].reshape(2*int_rank + 1))/nat[i]).astype(str)) for i in range(len(ftrs))]
    else:
        if int_rank == 0:
            tens = [str(ftrs[i].info[args.property]) for i in range(len(ftrs))]
        else:
            tens = [' '.join((np.array(ftrs[i].info[args.property].reshape(2*int_rank + 1))).astype(str)) for i in range(len(ftrs))]


    if kernels != None:
        kernels = kernels[0]
    if (kernels == None and sparsify == None):
        print("Either regular kernels or sparsification kernels must be specified!")
        sys.exit(0)
    if (kernels != None and sparsify != None):
        print("Either regular kernels or sparsification kernels must be specified (not both)!")
        sys.exit(0)

    # If a selection is given for the training set, read it in
    sel = args.select
    if (len(sel)==2):
        sel = [int(sel[0]),int(sel[1])]
    elif (len(sel)==1):
        # Read in an input file giving a training set
        sel = sel
    elif (len(sel) > 2):
        print("ERROR: too many arguments given to selection!")
        sys.exit(0)

    rdm = args.random

    if ((rdm == 0) & (len(sel)==0)):
        sel = [0,-1]

    jitter = [None]

    fractrain = ftr
    peratom = args.peratom
    prediction = True
    weights = args.weights
    mode = args.mode

    # Regression function
    def regression(log_reg_val):

        reg_val = 10.**log_reg_val
    
        # Do spherical regression, without environmental sparsification
        if (sparsify == None):
    
            # Read-in kernels
#            print "Loading kernel matrices..."
    
            kr = np.load(kernels)
            kernel = kr
    
            # Put tensor into float form
            spherical_tensor = np.array([i.split() for i in tens]).astype(float)
    
            int_rank = int(rank[-1])

            abs_error = sagpr_utils.do_sagpr_spherical(kernel,spherical_tensor,reg_val,rank_str=str(rank),nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=prediction,get_meantrain=True,mode=mode,wfile=weights,fnames=[args.features,kernels],jitter=jitter[0],get_rmse=True,verbose=False)
            return abs_error
    
        # Do spherical regression, with environmental sparsification
        else:
    
            if (len(sparsify) != 2):
                print("ERROR: two kernels must be specified!")
                sys.exit(0)
    
            # We want to sparsify on the rows of the kernel, so we're going to load in the sparsification kernels
    
            kr1 = np.load(sparsify[0])
            kr2 = np.load(sparsify[1])
            nN = len(kr1)
            nM = len(kr1[0])
            if (len(kr2) != nM or len(kr2[0]) != nM):
                print("ERROR: kernel matrices have incorrect dimensions!")
                sys.exit(0)
            kernel = [kr1,kr2]
    
            int_rank = int(rank[-1])
    
            # If we have chosen to do prediction, we have to split the kernel at this point into training and testing kernels
            if (prediction):
                if (len(sel)==1):
                    training_set = np.load(sel[0])
                elif (len(sel)==2):
                    training_set = list(range(sel[0],sel[1]))
                elif (rdm!=0):
                    training_set = list(range(nN))
                    random.shuffle(training_set)
                    training_set = training_set[:rdm]
                else:
                    print("ERROR: you have asked for prediction but have not specified a training set!")
                    sys.exit(0)
                # Now do the splitting of the kernel
                test_set = np.setdiff1d(list(range(nN)),training_set)
                ktr = np.array([[kernel[0][i,j] for j in range(len(kernel[0][0]))] for i in training_set]).astype(float)
                kte = np.array([[kernel[0][i,j] for j in range(len(kernel[0][0]))] for i in test_set]).astype(float)
                kernel[0] = ktr
    
            # We have loaded in the kernels, so we now combine these to get a lower-rank matrix
            # Kmn Knm
            if (len(np.shape(kernel[0])) != 2):
                sz = len(kernel[0][0,0])
                kernel[0] = kernel[0].transpose(0,2,1,3).reshape(-1,nM*sz)
                sparse_kernel = np.dot(kernel[0].T,kernel[0]).reshape(nM,sz,nM,sz).transpose(0,2,1,3)
            else:
                sparse_kernel = np.dot(kernel[0].T,kernel[0])
    
            # Regularization matrices
            if (len(np.shape(kernel[1])) != 2):
                sz = len(kernel[1][0,0])
                kernel[1] = kernel[1].transpose(0,2,1,3).reshape(-1,nM*sz)
            reg_matr = kernel[1]
    
            # The property must also be transformed
            fl_tens = np.zeros((len(tens),len(np.array(tens[0].split()).astype(float))),dtype=float)
            for i in range(len(tens)):
                fl_tens[i,:] = np.array(tens[i].split()).astype(float)
            # If we want to do prediction, split up the properties as well
            if (prediction):
                ptr = np.array([fl_tens[i] for i in training_set]).astype(float)
                pte = np.array([fl_tens[i] for i in test_set]).astype(float)
                nattest = np.array([nat[i] for i in test_set]).astype(int)
                fl_tens = ptr
            # Subtract the mean, if applicable
            meantrain = 0.0
            if (int_rank==0):
                meantrain = np.mean(fl_tens)
                fl_tens -= meantrain
            # Kmn alpha
            spherical_tensor = np.dot(kernel[0].T,np.reshape(fl_tens,np.size(fl_tens)))
            if (int_rank == 0):
                spherical_tensor = np.array([spherical_tensor[i] for i in range(len(spherical_tensor))])
    
            # Put tensor into float form
            spherical_tensor = np.split(spherical_tensor,len(spherical_tensor)/(2*int_rank+1))
    
            # Pass these matrices to the SA-GPR routine
            sagpr_utils.do_sagpr_spherical(sparse_kernel,spherical_tensor,reg_val,rank_str=str(rank),nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=False,reg_matr=reg_matr,get_meantrain=False,mode=mode,wfile=weights,fnames=[args.features,sparsify],jitter=jitter[0],verbose=False)
    
            # Deal with meantrain: put this into the weights so that we can do prediction
            if (int_rank == 0):
                wts = np.load(weights + "_" + str(rank) + ".npy")
                wts[5] = meantrain
                np.save(weights + "_" + str(rank) + ".npy",wts)
    
            # If we have asked for prediction here, get predictions for the testing set we created
    
            # Multiply weights by kernel to get predictions, reshaping kernel if necessary
            if (int_rank>0):
                shp = np.shape(kte)
                kte = kte.transpose(0,2,1,3).reshape(shp[0]*shp[2],shp[1]*shp[3])
                wts = np.load(weights + "_" + str(rank) + ".npy")
            pred = np.dot(kte,wts[4])
            if (int_rank==0):
                pred += meantrain
                pte = pte.reshape(len(pte))
                if peratom:
                    corrfile = open("prediction_L" + str(int_rank) + ".txt","w")
                    for i in range(len(pred)):
                        print(pte[i]*nattest[i],"  ",pred[i]*nattest[i],"  ",nattest[i], file=corrfile)
                else:
                    corrfile = open("prediction_L" + str(int_rank) + ".txt","w")
                    for i in range(len(pred)):
                        print(pte[i],"  ",pred[i], file=corrfile)
                # Accumulate errors
                intrins_dev = np.std(ptr)**2
                abs_error = 0.0
                for i in range(len(pte)):
                    abs_error += (pte[i] - pred[i])**2
                abs_error /= len(pte)
            else:
                pred = np.split(pred,len(pte))
                if peratom:
                    corrfile = open("prediction_L" + str(int_rank) + ".txt","w")
                    for i in range(len(pred)):
                        print(' '.join(str(e) for e in list(np.array(pte[i])*nattest[i])),"  ",' '.join(str(e) for e in list(np.array(pred[i])*nattest[i])),"  ",nattest[i], file=corrfile)
                else:
                    corrfile = open("prediction_L" + str(int_rank) + ".txt","w")
                    for i in range(len(pred)):
                        print(' '.join(str(e) for e in list(np.array(pte[i]))),"  ",' '.join(str(e) for e in list(np.array(pred[i]))), file=corrfile)
                intrins_dev = np.std(ptr)**2
                intrins_dev = 0.0
                abs_error = 0.0
                for i in range(len(ptr)):
                    intrins_dev += np.linalg.norm(ptr[i])**2
                for i in range(len(pte)):
                    abs_error += np.linalg.norm(pred[i]-pte[i])**2
                intrins_dev /= len(ptr)
                abs_error /= len(pte)

            return abs_error

    init_simplex = np.reshape([np.log10(reg[0]),np.log10(reg[1])],(2,1))

    # Do Nelder-Mead optimization to find the best regularization
    nm_out = optimize.minimize(regression,x0=init_simplex[0],method='Nelder-Mead',tol=func_tol,options={'initial_simplex':init_simplex,'disp':False,'fatol':func_tol})

    print(nm_out)

    out_reg = 10.**nm_out['x'][0]
    formatted_reg = '{:.{prec}e}'.format(out_reg, prec=args.precision)

    print()
    print("Final regularization:")
    print("---------------------")
    print()
    print(formatted_reg)

    print()
    print("Final function value:")
    print("---------------------")
    print()
    print(regression(np.log10(float(formatted_reg))))

if __name__=="__main__":
    main()
