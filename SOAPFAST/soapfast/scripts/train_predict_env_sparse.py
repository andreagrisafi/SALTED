#!/usr/bin/env python

import argparse
import sys,os
import numpy as np
from ase.io import read,write
from get_training_set import get_training_set
from get_atomic_power_spectrum import get_atomic_power_spectrum
from do_fps import generate_FPS
from apply_fps import apply_FPS
sys.path.insert(1,os.path.join(sys.path[0],'..'))
from get_kernel import get_kernel
import random
import utils.sagpr_utils

def do_sparse_learn_predict(PS,frames,reg,klist,prop,scale = [],n_env = [1000],initial = [-1],ntrain = -1,setmode = 'seq',regmode = 'pinv', threshold = 1e-8,infile = '',outfile = '',zeta = 1,peratom = False,spherical=False):

    # If we have too few of any input arguments supplied, fill the rest of the array based on what we already have
    if (len(n_env) < len(PS)):
        for i in range(len(n_env),len(PS)):
            n_env.append(n_env[-1])
    if (len(initial) < len(PS)):
        for i in range(len(initial),len(PS)):
            initial.append(initial[-1])

    # If we have an L=0 property, spherical should be set to true
    if (len(PS)==1 and len(np.shape(PS[0]))==3):
        spherical = True

    # In several steps, generate an environmentally sparsified model and test it

    # Firstly, using the power spectrum generate a training and testing set
    if ((ntrain==-1) or (ntrain>len(PS[0]))):
        ntrain = len(PS[0])
    train_test_set   = get_training_set(PS,frames,scale=scale,ntrain=ntrain,mode=setmode,infile=infile)
    training_indices = train_test_set[2]
    testing_indices  = train_test_set[3]
    full_training    = []
    for i in range(len(PS)):
        full_training.append(train_test_set[6 + 2*i])
    full_testing     = []
    for i in range(len(PS)):
        full_testing.append(train_test_set[7 + 2*i])

    # Next, get an atomic power spectrum
    lam = np.zeros(len(PS),dtype=int)
    for i in range(len(PS)):
        if (len(np.shape(PS[i])) == 3):
            lam[i] = 0
        else:
            lam[i] = (len(PS[i][0,0]) - 1)/2
    if (lam[0] != 0):
        print("ERROR: the first power spectrum should have lambda=0 for the purposes of determining FPS ordering and building nonlinear kernels!")
        sys.exit(0)
    frame_train = train_test_set[4]
    power_atomic = [get_atomic_power_spectrum(full_training[i],frame_train,lam=lam[i]) for i in range(len(PS))]

    # Retain the specified number of environments
    reordered_train = []
    if spherical:
        FPS_details = generate_FPS(power_atomic[-1],nsparse=n_env[-1],initial=initial[-1])
    else:
        FPS_details = generate_FPS(power_atomic[0],nsparse=n_env[0],initial=initial[0])
    for i in range(len(PS)):
        # Sparsify this atomic power spectrum
        reordered   = apply_FPS(power_atomic[i],FPS_details)
        reordered_train.append(reordered)

    # Build all the appropriate kernels
    K_MM = []
    K_NM = []
    K_TT = []
    scale_train = np.array([scale[i] for i in training_indices])
    scale_test  = np.array([scale[i] for i in testing_indices])
    for i in range(len(PS)):
        scale_atomic = np.array([1 for j in range(len(reordered_train[i]))])
        if (lam[i] == 0):
            K_MM.append(get_kernel([reordered_train[i],reordered_train[i]],zeta=zeta))
            K_NM.append(get_kernel([full_training[i],reordered_train[i]],scale=[scale_train,scale_atomic],zeta=zeta))
            K_TT.append(get_kernel([full_testing[i],reordered_train[i]],scale=[scale_test,scale_atomic],zeta=zeta))
        else:
            K_MM.append(get_kernel([reordered_train[i],reordered_train[i]],PS0=[reordered_train[0],reordered_train[0]],zeta=zeta))
            K_NM.append(get_kernel([full_training[i],reordered_train[i]],PS0=[full_training[0],reordered_train[0]],scale=[scale_train,scale_atomic],zeta=zeta))
            K_TT.append(get_kernel([full_testing[i],reordered_train[i]],PS0=[full_testing[0],reordered_train[0]],scale=[scale_test,scale_atomic],zeta=zeta))

    # Get the property training set and split it into tensor components
    rank = lam[-1]
    meantrain = [0 for i in range(len(lam))]
    if (not spherical):
        if peratom:
            if rank == 0:
                tens = [str(frame_train[i].info[prop]/scale_train[i]) for i in range(len(frame_train))]
            elif rank == 2:
                tens = [' '.join((np.concatenate(frame_train[i].info[prop])/scale_train[i]).astype(str)) for i in range(len(frame_train))]
            else:
                tens = [' '.join((np.array(frame_train[i].info[prop])/scale_train[i]).astype(str)) for i in range(len(frame_train))]
        else:
            if rank == 0:
                tens = [str(frame_train[i].info[prop]) for i in range(len(frame_train))]
            elif rank == 2:
                tens = [' '.join(np.concatenate(frame_train[i].info[prop]).astype(str)) for i in range(len(frame_train))]
            else:
                tens = [' '.join(np.array(frame_train[i].info[prop]).astype(str)) for i in range(len(frame_train))]
        if (len(lam) > 1):
            [spherical_tensor,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = utils.sagpr_utils.get_spherical_tensor_components(tens,rank,threshold)
        else:
            spherical_tensor = [np.array(tens,dtype=float)]
            keep_list = [0]
            degen = [1]
    else:
        if peratom:
            if rank == 0:
                tens = [str(frame_train[i].info[prop]/scale_train[i]) for i in range(len(frame_train))]
            else:
                tens = [' '.join((np.array(frame_train[i].info[prop].reshape(2*rank + 1))/scale_train[i]).astype(str)) for i in range(len(frame_train))]
        else:
            if rank == 0:
                tens = [str(frame_train[i].info[prop]) for i in range(len(frame_train))]
            else:
                tens = [' '.join((np.array(frame_train[i].info[prop].reshape(2*rank + 1))).astype(str)) for i in range(len(frame_train))]
        # Put tensor into float form
        if (len(lam) > 1):
            spherical_tensor = np.array([i.split() for i in tens]).astype(float)
            spherical_tensor = [np.reshape(spherical_tensor,np.size(spherical_tensor))]
        else:
            spherical_tensor = [np.array(tens,dtype=float)]

    if (len(spherical_tensor) != len(klist)):
        print("ERROR: number of kernels in the kernel list (%i) must match the number of spherical tensor components (%i)!"%(len(klist),len(spherical_tensor)))

    # Do the regression
    if (len(reg) < len(klist)):
        for i in range(len(initial),len(klist)):
            reg.append(reg[-1])
    weights = []
    for i in range(len(klist)):
        print("Doing regression for L=%i"%lam[klist[i]])
        if (lam[klist[i]] == 0):
            # Get Kmn Knm
            K_MN_K_NM = np.dot(K_NM[klist[i]].T,K_NM[klist[i]])
            # Get regularization matrix
            reg_matr = K_MM[klist[i]]
            # Transform spherical property
            spherical_tensor[i]  = np.real(spherical_tensor[i]).astype(float)
            meantrain[i]         = np.mean(spherical_tensor[i])
            spherical_tensor[i] -= meantrain[i]
            spherical_tensor[i]  = np.dot(K_NM[klist[i]].T,spherical_tensor[i])
        else:
            # Get Kmn Knm
            sz  = len(K_NM[klist[i]][0,0])
            nM  = len(K_NM[klist[i]][0])
            knm = K_NM[klist[i]].transpose(0,2,1,3).reshape(-1,nM*sz)
            K_MN_K_NM = np.dot(knm.T,knm).reshape(nM,sz,nM,sz).transpose(0,2,1,3)
            # Get regularization matrix
            reg_matr = K_MM[klist[i]].transpose(0,2,1,3).reshape(-1,nM*sz)
            # Transform spherical property
            knm = K_NM[klist[i]].transpose(0,2,1,3).reshape(-1,nM*sz)
            spherical_tensor[i] = np.dot(knm.T,spherical_tensor[i])
            spherical_tensor[i] = np.split(spherical_tensor[i],len(spherical_tensor[i])/(2*lam[klist[i]] + 1))
        # Pass these to SA-GPR
        if (not spherical):
            wt = utils.sagpr_utils.do_sagpr_spherical(K_MN_K_NM,spherical_tensor[i],reg[i],rank_str=''.join(map(str,keep_list[i][1:])),nat=scale_train,peratom=peratom,reg_matr=reg_matr,get_meantrain=False,mode=regmode,wfile='')
        else:
            wt = utils.sagpr_utils.do_sagpr_spherical(K_MN_K_NM,spherical_tensor[i],reg[i],rank_str=str(lam[klist[i]]),nat=scale_train,peratom=peratom,reg_matr=reg_matr,get_meantrain=False,mode=regmode,wfile='')
        if (lam[klist[i]]==0):
            wt[5] = meantrain[i]
        weights.append(wt)

    # Do the prediction
    outvec = []
    for i in range(len(klist)):
        if (not spherical):
            str_rank = ''.join(map(str,keep_list[i][1:]))
            if (str_rank == ''):
                str_rank = ''.join(map(str,keep_list[i]))
        else:
            str_rank = str(lam[klist[i]])
        ov = utils.sagpr_utils.do_prediction_spherical(K_TT[klist[i]],rank_str=str_rank,weightfile='',outfile='',weight_array=weights[i])
        outvec.append(ov)
    ns = int(len(outvec[0]) / (2*lam[klist[0]]+1))
    if (not spherical):
        if (len(lam)>1):
            predcart = utils.regression_utils.convert_spherical_to_cartesian(outvec,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)
        else:
            predcart = outvec

    # Print out errors
    frame_test = train_test_set[5]
    if (not spherical):
        if peratom:
            if rank == 0:
                testtens = [str(frame_test[i].info[prop]/scale_test[i]) for i in range(len(frame_test))]
            elif rank == 2:
                testtens = [' '.join((np.concatenate(frame_test[i].info[prop])/scale_test[i]).astype(str)) for i in range(len(frame_test))]
            else:
                testtens = [' '.join((np.array(frame_test[i].info[prop])/scale_test[i]).astype(str)) for i in range(len(frame_test))]
        else:
            if rank == 0:
                testtens = [str(frame_test[i].info[prop]) for i in range(len(frame_test))]
            elif rank == 2:
                testtens = [' '.join(np.concatenate(frame_test[i].info[prop]).astype(str)) for i in range(len(frame_test))]
            else:
                testtens = [' '.join(np.array(frame_test[i].info[prop]).astype(str)) for i in range(len(frame_test))]
        # Get spherical components
        if (len(lam) > 1):
            [test_spherical_tensor,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = utils.sagpr_utils.get_spherical_tensor_components(testtens,rank,threshold)
            for i in range(len(klist)):
                if (lam[klist[i]] == 0):
                    for j in range(len(test_spherical_tensor[i])):
                        test_spherical_tensor[i][j] = test_spherical_tensor[i][j][0]
        else:
            test_spherical_tensor = [np.array(testtens,dtype=float)]
        testcart = np.concatenate([np.array(testtens[i].split()).astype(float) for i in range(len(testtens))])
    # We are doing a spherical prediction
    else:
        if peratom:
            if rank == 0:
                testtens = [str(frame_test[i].info[prop]/scale_test[i]) for i in range(len(frame_test))]
            else:
                testtens = [' '.join((np.array(frame_test[i].info[prop].reshape(2*rank + 1))/scale_test[i]).astype(str)) for i in range(len(frame_test))]
        else:
            if rank == 0:
                testtens = [str(frame_test[i].info[prop]) for i in range(len(frame_test))]
            else:
                testtens = [' '.join((np.array(frame_test[i].info[prop].reshape(2*rank + 1))).astype(str)) for i in range(len(frame_test))]
        # Put tensor into float form
        if (len(lam) > 1):
            test_spherical_tensor = np.array([i.split() for i in testtens]).astype(float)
            test_spherical_tensor = [np.reshape(test_spherical_tensor,np.size(test_spherical_tensor))]
        else:
            test_spherical_tensor = [np.array(testtens,dtype=float)]
    
    print()
    print("Prediction Errors:")
    print("==================")
    print("Testing data points:  %i"%len(testing_indices))
    print("Training data points: %i"%len(training_indices))
    for i in range(len(klist)):
        intrins_dev = np.std(test_spherical_tensor[i])**2
        abs_error = np.sum((outvec[i] - test_spherical_tensor[i])**2) / ns
        print()
        print("Errors for L=%i"%lam[klist[i]])
        print("--------------")
        print("ST DEV   = %.4f"%np.sqrt(intrins_dev))
        print("ABS RMSE = %.4f"%np.sqrt(abs_error))
        print("RMSE     = %.4f %%"%(100. * np.sqrt(np.abs(abs_error / intrins_dev))))
    if (not spherical):
        intrins_dev = np.std(testcart)**2
        abs_error = np.sum((predcart - testcart)**2) / ns
        print()
        print("Cartesian errors")
        print("----------------")
        print("ST DEV   = %.4f"%np.sqrt(intrins_dev))
        print("ABS RMSE = %.4f"%np.sqrt(abs_error))
        print("RMSE     = %.4f %%"%(100. * np.sqrt(np.abs(abs_error / intrins_dev))))

    # If desired, also print out weights and kernels
    if outfile != '':
        # Print out the FPS details, environmental power spectra and all kernels
        np.save(outfile + "_fps_details.npy",FPS_details)
        for i in range(len(reordered_train)):
            np.save(outfile + "_power_spectrum_" + str(i) + ".npy",reordered_train[i])
        for i in range(len(klist)):
            np.save(outfile + "_kernel_MM_" + str(klist[i]) + ".npy",K_MM[klist[i]])
            np.save(outfile + "_kernel_NM_" + str(klist[i]) + ".npy",K_NM[klist[i]])
            np.save(outfile + "_kernel_TT_" + str(klist[i]) + ".npy",K_TT[klist[i]])

    # Whether or not we print out all information, we definitely want the weights and the predictions for the testing set
    if outfile != '':
        outfile = '_' + outfile
    for i in range(len(klist)):
        np.save('weights' + outfile + '_' + str(lam[i]) + '.npy',np.array(weights[i],dtype=object))
        if (not spherical):
            predfile = open('prediction' + outfile + '_L' + ''.join(map(str,keep_list[i][1:])) + '.txt','w')
        else:
            predfile = open('prediction' + outfile + '_L' + str(lam[klist[i]]) + '.txt','w')
        for j in range(int(len(test_spherical_tensor[i]) / (2*lam[klist[i]] + 1))):
            if peratom:
                print(' '.join(str(e) for e in list(np.split(np.array(test_spherical_tensor[i]),ns)[j]*scale_test[j])),' ',' '.join(str(e) for e in list(np.split(np.array(outvec[i]),ns)[j]*scale_test[j])),' ',scale_test[j], file=predfile)
            else:
                print(' '.join(str(e) for e in list(np.split(np.array(test_spherical_tensor[i]),ns)[j])),' ',' '.join(str(e) for e in list(np.split(np.array(outvec[i]),ns)[j])), file=predfile)

###########################################################################################################

def main():

    parser = argparse.ArgumentParser(description="Full training and prediction for environmental sparsification")
    parser.add_argument("-p",     "--power",                      required=True, nargs='+',                     help="Power spectrum file")
    parser.add_argument("-fr",    "--frames",                     required=True,                                help="Atomic coordinates")
    parser.add_argument("-s",     "--scaling",                    default=[],                                   help="Scaling file")
    parser.add_argument("-i",     "--initial",        type=int,   default=[-1], nargs='+',                      help="Initial row")
    parser.add_argument("-sm",    "--setmode",                    choices=['seq','rdm','input'], default="seq", help="Mode for choosing the training set")
    parser.add_argument("-n",     "--ntrain",         type=int,   default=-1,                                   help="Number of training points")
    parser.add_argument("-e",     "--env",            type=int,   default=[1000], nargs='+',                    help="Number of environments")
    parser.add_argument("-f",     "--infile",                     default='',                                   help="Input file for training set")
    parser.add_argument("-o",     "--ofile",                      default='',                                   help="Output file prefix")
    parser.add_argument("-z",     "--zeta",           type=int,   default=1,                                    help="Nonlinearity parameter")
    parser.add_argument("-k",     "--klist",          type=int,   default=[0], nargs='+',                       help="List of kernels to use for regression")
    parser.add_argument("-pr",    "--property",                   required=True,                                help="Property on which to carry out regression")
    parser.add_argument("-reg",   "--regularization", type=float, required=True, nargs='+',                     help="Regularization parameters")
    parser.add_argument("-perat", "--peratom",                    action='store_true',                          help="Predict per-atom properties?")
    parser.add_argument("-t",     "--threshold",      type=float, default=1e-8,                                 help="Threshold value for spherical component zeroing")
    parser.add_argument("-rm",    "--regmode",        type=str,   choices=['solve','pinv'], default='pinv',     help="Mode to use for inversion of kernel matrices")
    parser.add_argument("-sp",    "--spherical",                  action='store_true',                          help="Learn a spherical property")
    args = parser.parse_args()

    PS = [np.load(args.power[i]) for i in range(len(args.power))]
    setmode   = args.setmode
    infile    = args.infile
    initial   = args.initial
    outfile   = args.ofile
    ntrain    = args.ntrain
    env       = args.env
    zeta      = args.zeta
    klist     = args.klist
    prop      = args.property
    reg       = args.regularization
    perat     = args.peratom
    regmode   = args.regmode
    threshold = args.threshold
    spherical = args.spherical
    if ((setmode == 'input') and (infile == '')):
        print("ERROR: an input file must be specified!")
        sys.exit(0)

    if (len(env) < len(PS)):
        for i in range(len(env),len(PS)):
            env.append(env[-1])
    if (len(initial) < len(PS)):
        for i in range(len(initial),len(PS)):
            initial.append(initial[-1])
    if (len(reg) < len(klist)):
        for i in range(len(initial),len(klist)):
            reg.append(reg[-1])

    if (args.scaling == ''):
        scale = np.array([1 for i in range(nrow)])
    else:
        scale = np.load(args.scaling)

    frames = read(args.frames,':')

    do_sparse_learn_predict(PS,frames,reg,klist,prop,scale=scale,initial=initial,n_env=env,ntrain=ntrain,setmode=setmode,regmode=regmode,threshold=threshold,infile=infile,outfile=outfile,zeta=zeta,peratom=perat,spherical=spherical)

if __name__=="__main__":
    main()
