#!/usr/bin/python

import numpy as np
import sys
from random import shuffle
import scipy.linalg
from .regression_utils import *

###############################################################################################################################

def get_weights(ktrain,vtrain_part,mode,jitter):

    # Decide how to calculate regression weights
    if (mode == 'pinv'):
        return np.dot(np.linalg.pinv(ktrain),vtrain_part)
    elif (mode == 'solve'):
        if (jitter == None):
            return scipy.linalg.solve(ktrain,vtrain_part)
        else:
            if (jitter == 'CHOOSE'):
                # Choose the smallest jitter term that makes the matrix full-rank
                ln = len(ktrain)
                jt = 1.0e-12
                for mm in range(25):
                    jt *= 10
                    ktr = ktrain + jt*np.eye(ln)
                    if (np.linalg.matrix_rank(ktr)==ln):
                        break
                print("Jitter term %e gives full-rank matrix"%jt)
                ktrain = ktr
            else:
                ln = len(ktrain)
                jt = float(jitter)
                ktrain = ktrain + jt*np.eye(ln)
                print("With jitter term %e matrix rank is %i, compared to ideal value of %i."%(jt,np.linalg.matrix_rank(ktrain),ln))
            return scipy.linalg.solve(ktrain,vtrain_part)
                
    else:
        print("INVALID WEIGHTS SOLVING MODE: " + mode)
        sys.exit(0)

###############################################################################################################################

def do_sagpr_spherical(kernel,tens,reg,rank_str='',nat=[],fractrain=1.0,rdm=0,sel=[-1,-1],peratom=False,prediction=False,reg_matr=[],get_meantrain=True,mode='solve',wfile='weights',fnames=['',''],jitter=None,get_rmse=False,verbose=True):

    # initialize regression
    if (len(np.shape(kernel))==2):
        degen = 1
        lval  = 0
    else:
        degen = len(kernel[0,0])
        lval  = (degen-1)/2
    if (rank_str == ''):
        rank_str = str(lval)

    if (nat == []):
        nat = [1 for i in range(len(tens))]

    if (sel == [0,-1]) or (sel == [0,0]):
        sel = [0,len(tens)]

    if (len(sel)==1):
        sel = ['file',np.load(sel[0])]

    # Get a list of members of the training and testing sets
    ndata = len(tens)
    [ns,nt,ntmax,trrange,terange] = shuffle_data(ndata,sel,rdm,fractrain)
   
    # If we are doing sparsification, set the training range equal to the entire transformed training set
    if (reg_matr != []):
        trrange = list(range(len(tens)))
        terange = []
        nat = [1 for i in trrange]

    # Partition properties and kernel for training and testing
    [vtrain,vtest,ktr,kte,nattrain,nattest] = partition_kernels_properties_spherical(tens,kernel,trrange,terange,nat)
    vtrain_part = vtrain.reshape(np.size(vtrain))
    vtest_part  = vtest.reshape(np.size(vtest))

    # Subtract the mean if L=0
    meantrain = 0.0
    if degen==1:
        vtrain_part  = np.real(vtrain_part).astype(float)
        if get_meantrain:
            meantrain    = np.mean(vtrain_part)
            vtrain_part -= meantrain

    # Build training kernels
    [ktrain,ktrainpred] = build_training_kernel(len(ktr),degen,ktr,reg,reg_matr)

    # Invert training kernels
    invktrvec = get_weights(ktrain,vtrain_part,mode,jitter)

    # Print the details of the model to an external file
    # We want to know all important details: the filename and kernel file, the training set, the lambda value, the weights and if relevant the average value.
    weights = [lval,fnames[0],fnames[1],trrange,invktrvec]
    if (degen==1):
        weights.append(meantrain)
    if wfile != '':
        np.save(wfile + "_" + rank_str + ".npy",np.array(weights,dtype=object))
    else:
        return weights

    # Do we want to do prediction?

    if (prediction and ns>0):

        # Build testing kernels
        ktest = build_testing_kernel(ns,nt,degen,kte)

        # Predict on test data set
        outvec = np.dot(ktest,invktrvec)
        if degen==1:
            outvec += meantrain

        # Accumulate errors
        intrins_dev = np.std(vtest_part)**2
        abs_error = np.sum((outvec-vtest_part)**2)/(degen*ns)

        if peratom:
            corrfile = open("prediction_L" + rank_str + ".txt","w")
            for i in range(ns):
                print(' '.join(str(e) for e in list(np.array(vtest[i])*nattest[i])),"  ",' '.join(str(e) for e in list(np.split(outvec,ns)[i]*nattest[i])),"  ",str(nattest[i]), file=corrfile)
        else:
            corrfile = open("prediction_L" + rank_str + ".txt","w")
            for i in range(ns):
                print(' '.join(str(e) for e in vtest[i]), "  ", ' '.join(str(e) for e in list(np.split(outvec,ns)[i])), file=corrfile)

        # Print out errors
        if verbose:
            print("")
            print("testing data points: ", ns)
            print("training data points: ", nt)
            print("--------------------------------")
            print("RESULTS FOR L=%i MODULI (lambda=%f)"%(lval,reg))
            print("-----------------------------------------------------")
            print("STD", np.sqrt(intrins_dev))
            print("ABS RMSE", np.sqrt(abs_error))
            print("RMSE = %.4f %%"%(100. * np.sqrt(np.abs(abs_error / intrins_dev))))

        if get_rmse:
            return np.sqrt(abs_error)
        else:
            return [outvec,np.concatenate(vtest),nattest]

    else:

        return [None,None,None]

###############################################################################################################################

def get_spherical_tensor_components(tens,rank,threshold):

    # Get spherical components from input tensor
    [all_CS,keep_cols,all_sym] = get_cartesian_to_spherical(rank)
    CS = all_CS[-1]
    tensor = [np.array(tens[i].split()).astype(float) for i in range(len(tens))]
    [spherical_components,keep_list,CR,degen,lin_dep_list,sym_list] = get_spherical_components(tensor,CS,threshold,keep_cols,all_sym)

    return [spherical_components,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list]

###############################################################################################################################

def do_prediction_spherical(ktest,rank_str='',weightfile='weights',outfile='prediction',weight_array=[]):

    # initialize regression
    if (len(np.shape(ktest)) == 2):
        degen = 1
        lval = 0
    else:
        degen = len(ktest[0,0])
        lval = (degen-1)/2

    if (rank_str == ''):
        rank_str = str(lval)

    ns = len(ktest)
    nt = len(ktest[0])

    # Get weights
    if (weightfile != ''):
        weights = np.load(weightfile + "_" + rank_str + ".npy",allow_pickle=True)
    elif weight_array != []:
        weights = weight_array
    else:
        print("ERROR: weights must be given!")
        sys.exit(0)

    # Unpack the array containing the pre-calculated model
    meantrain = 0.0
    invktrvec = weights[4]
    if degen == 1:
        meantrain = weights[5]

    # Reshape testing kernel if necessary
    if degen>1:
        ktest = ktest.transpose(0,2,1,3).reshape(-1,nt*degen)

    # Predict on test data set
    outvec = np.dot(ktest,invktrvec)
    if degen==1:
        outvec += meantrain

    # Print out predictions
    if outfile != '':
        corrfile = open(outfile + "_L" + rank_str + ".txt","w")
        for i in range(ns):
            print(' '.join(str(e) for e in list(np.split(outvec,ns)[i])), file=corrfile)

    return outvec
