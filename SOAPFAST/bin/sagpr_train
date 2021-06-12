#!/usr/bin/env python

from utils import parsing,regression_utils,sagpr_utils
import scipy.linalg
import random
import sys
import numpy as np

###############################################################################################################################

def main():

    # This is a wrapper that calls python scripts to do SA-GPR with pre-built L-SOAP kernels.
    
    # Parse input arguments
    args = parsing.add_command_line_arguments_learn("SA-GPR")
    [reg,fractrain,tens,kernels,sel,rdm,rank,nat,peratom,prediction,weights,sparsify,mode,threshold,jitter] = parsing.set_variable_values_learn(args)
    
    if (args.spherical == False):
    
        # Do full Cartesian regression, without environmental sparsification
        if (sparsify == None):
        
            # Read-in kernels
            print("Loading kernel matrices...")
            
            kernel = []
            for k in range(len(kernels)):
                kr = np.load(kernels[k])
                kernel.append(kr)
            
            print("...Kernels loaded.")
    
            # Get spherical components
            [spherical_tensor,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components(tens,rank,threshold)
            for l in range(len(degen)):
                if (degen[l] != 1): 
                    spherical_tensor[l] = np.split(spherical_tensor[l],len(spherical_tensor[l])/degen[l])

            outvec = []
            tstvec = []
            for l in range(len(degen)):
                # Do regression for each spherical tensor component
                lval = keep_list[l][-1]
                str_rank = ''.join(map(str,keep_list[l][1:]))
                if (str_rank == ''):
                    str_rank = ''.join(map(str,keep_list[l]))
                [ov, tv, na] = sagpr_utils.do_sagpr_spherical(kernel[l],spherical_tensor[l],reg[l],rank_str=str_rank,nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=prediction,get_meantrain=True,mode=mode,wfile=weights,fnames=[args.features,kernels[l]],jitter=jitter[l])
                outvec.append(ov)
                tstvec.append(tv)
            nattest = na
    
            # If we wanted to do the predictions, then put together the predicted spherical tensors
            if (prediction):
                ns = int(len(outvec[0])/degen[0])
                predcart  = regression_utils.convert_spherical_to_cartesian(outvec,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)
                testcart  = regression_utils.convert_spherical_to_cartesian(tstvec,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)
    
                # Print out predictions
                if peratom:
                    corrfile = open("prediction_cartesian.txt","w")
                    for i in range(ns):
                        print(' '.join(str(e) for e in list(np.split(testcart,ns)[i]*nattest[i])),"  ", ' '.join(str(e) for e in list(np.split(predcart,ns)[i]*nattest[i])),"  ",str(nattest[i]), file=corrfile)
                    corrfile.close()
                else:
                    corrfile = open("prediction_cartesian.txt","w")
                    for i in range(ns):
                        print(' '.join(str(e) for e in list(np.split(testcart,ns)[i])),"  ", ' '.join(str(e) for e in list(np.split(predcart,ns)[i])), file=corrfile)
                    corrfile.close()
    
        # Do full Cartesian regression, with environmental sparsification
        else:
        
            # We want to sparsify on the rows of the kernel, so we're going to load in the sparsification kernels
            print("Loading sparsification kernels...")
            kernel = []
            reg_matr = []
        
            # Check that the dimensions of these kernels are as they should be
            # There should be pairs of kernels, with the appropriate dimensions
            nN = 0
            nM = 0
            for k in range(int(len(sparsify)/2)):
                kr1 = np.load(sparsify[2*k])
                kr2 = np.load(sparsify[2*k+1])
                if (nN != 0):
                    if (nN != len(kr1) or nM != len(kr1[0])):
                        print("ERROR: kernel matrices have different dimensions!")
                        sys.exit(0)
                else:
                    nN = len(kr1)
                    nM = len(kr1[0])
                if (len(kr2) != nM or len(kr2[0]) != nM):
                    print("ERROR: kernel matrices have incorrect dimensions!")
                    sys.exit(0)
                kernel.append(kr1)
                kernel.append(kr2)
        
            print("...Kernels loaded.")

            # If we have chosen to do prediction, we have to split the kernels at this point into training and testing kernels
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
                # Now split the kernels
                test_set = np.setdiff1d(list(range(nN)),training_set)
                ktr = []
                kte = []
                for k in range(int(len(sparsify)/2)):
                    ktr.append(np.array([[kernel[2*k][i,j] for j in range(len(kernel[2*k][0]))] for i in training_set]).astype(float))
                    kte.append(np.array([[kernel[2*k][i,j] for j in range(len(kernel[2*k][0]))] for i in test_set]).astype(float))
                    kernel[2*k] = ktr[-1]
        
            # We have loaded in the kernels, so now combine these firstly to get a lower-rank matrix
            sparse_kernel = []
            # Kmn Knm
        
            for k in range(int(len(sparsify)/2)):
                if (len(np.shape(kernel[2*k])) != 2):
                    sz = len(kernel[2*k][0,0])
                    kernel[2*k] = kernel[2*k].transpose(0,2,1,3).reshape(-1,nM*sz)
                    sparse_kernel.append(np.dot(kernel[2*k].T,kernel[2*k]).reshape(nM,sz,nM,sz).transpose(0,2,1,3))
                else:
                    sparse_kernel.append(np.dot(kernel[2*k].T,kernel[2*k]))
        
            # Regularization matrices
            reg_matr = []
            for k in range(int(len(sparsify)/2)):
                if (len(np.shape(kernel[2*k+1])) != 2):
                    sz = len(kernel[2*k+1][0,0])
                    kernel[2*k+1] = kernel[2*k+1].transpose(0,2,1,3).reshape(-1,nM*sz)
                reg_matr.append(kernel[2*k+1])
        
            # Next, also transform the properties; first, we must convert these to spherical components
            [spherical_tensor,degen,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list] = sagpr_utils.get_spherical_tensor_components(tens,rank,threshold)

            # If we want to do prediction, split up the properties as well
            if (prediction):
                ptr = []
                pte = []
                for i in range(len(degen)):
                    ptr.append(np.reshape(np.array([spherical_tensor[i][degen[i]*j:degen[i]*(j+1)] for j in training_set]).astype(float),len(training_set)*degen[i]))
                    pte.append(np.reshape(np.array([spherical_tensor[i][degen[i]*j:degen[i]*(j+1)] for j in test_set]).astype(float),len(test_set)*degen[i]))
                    nattest = np.array([nat[j] for j in test_set]).astype(int)
                    spherical_tensor[i] = ptr[-1]
    
            # Subtract the mean if there is one, as we don't want to transform this
            meantrain = [0 for i in range(len(degen))]
            for i in range(len(degen)):
                if degen[i]==1:
                    spherical_tensor[i]  = np.real(spherical_tensor[i]).astype(float)
                    meantrain[i]    = np.mean(spherical_tensor[i])
                    spherical_tensor[i] -= meantrain[i]
        
            # Apply the transformations
            # Kmn alpha
            for k in range(int(len(sparsify)/2)):
                spherical_tensor[k] = np.dot(kernel[2*k].T,spherical_tensor[k])
    
            for l in range(len(degen)):
                if (degen[l] != 1):
                    spherical_tensor[l] = np.split(spherical_tensor[l],len(spherical_tensor[l])/degen[l])
    
            # Now pass these to the SA-GPR routine
            for l in range(len(degen)):
                lval = keep_list[l][-1]
                if (len(keep_list[l])==1 and keep_list[l][-1]==1):
                    sagpr_utils.do_sagpr_spherical(sparse_kernel[l],spherical_tensor[l],reg[l],rank_str='1',nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=False,reg_matr=reg_matr[l],get_meantrain=False,mode=mode,wfile=weights,fnames=[args.features,sparsify[l]],jitter=jitter[l])
                else:
                    sagpr_utils.do_sagpr_spherical(sparse_kernel[l],spherical_tensor[l],reg[l],rank_str=''.join(map(str,keep_list[l][1:])),nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=False,reg_matr=reg_matr[l],get_meantrain=False,mode=mode,wfile=weights,fnames=[args.features,sparsify[l]],jitter=jitter[l])
    
            # Deal with meantrain: put this into the weights so that we can do prediction
            for i in range(len(degen)):
                str_rank = ''.join(map(str,keep_list[i][1:]))
                if (degen[i]==1):
                    wts = np.load(weights + "_" + str_rank + ".npy",allow_pickle=True)
                    wts[5] = meantrain[i]
                    np.save(weights + "_" + str_rank + ".npy",wts)

            # If we have asked for prediction here, get predictions for the testing set we created
            if (prediction):

                # First, get the predictions for each individual spherical order
                pred = []
                for i in range(len(degen)):
                    if (degen[i]>1):
                        shp = np.shape(kte[i])
                        kte[i] = kte[i].transpose(0,2,1,3).reshape(shp[0]*shp[2],shp[1]*shp[3])
                    str_rank = ''.join(map(str,keep_list[i][1:]))
                    if (str_rank == ''):
                        str_rank = ''.join(map(str,keep_list[i]))
                    wts = np.load(weights + "_" + str_rank + ".npy",allow_pickle=True)
                    pred.append(np.dot(kte[i],wts[4]))
                    if (degen[i]==1):
                        pred[-1] += meantrain[i]
                        pte[i] = pte[i].reshape(len(pte[i]))
                        # Print out predictions
                        if peratom:
                            corrfile = open("prediction_L" + str_rank + ".txt","w")
                            for j in range(len(pred[i])):
                                print(pte[i][j]*nattest[j],"  ",pred[i][j]*nattest[j],"  ",nattest[j], file=corrfile)
                            corrfile.close()
                        else:
                            corrfile = open("prediction_L" + str_rank + ".txt","w")
                            for j in range(len(pred)):
                                print(pte[i][j],"  ",pred[i][j], file=corrfile)
                            corrfile.close()
                        # Accumulate errors
                        intrins_dev = np.std(ptr[i])**2
                        abs_error = 0.0
                        for j in range(len(pte[i])):
                            abs_error += (pte[i][j] - pred[i][j])**2
                        abs_error /= len(pte[i])
                    else:
                        prediction = pred[i].reshape(len(test_set),degen[i])
                        comparison = pte[i].reshape(len(test_set),degen[i])
                        # Print out predictions
                        if peratom:
                            corrfile = open("prediction_L" + str_rank + ".txt","w")
                            for j in range(len(prediction)):
                                print(' '.join(str(e) for e in list(np.array(comparison[j])*nattest[j])),"  ",' '.join(str(e) for e in list(np.array(prediction[j])*nattest[j])),"  ",nattest[j], file=corrfile)
                            corrfile.close()
                        else:
                            corrfile = open("prediction_L" + str_rank + ".txt","w")
                            for j in range(len(prediction)):
                                print(' '.join(str(e) for e in list(np.array(comparison[j]))),"  ",' '.join(str(e) for e in list(np.array(prediction[j]))), file=corrfile)
                            corrfile.close()
                        # Accumulate errors
                        intrins_dev=0.0
                        abs_error=0.0
                        training = ptr[i].reshape(len(training_set),degen[i])
                        for j in range(len(training)):
                            intrins_dev += np.linalg.norm(training[j])**2
                        for j in range(len(comparison)):
                            abs_error += np.linalg.norm(comparison[j]-prediction[j])**2
                        intrins_dev /= len(training)
                        abs_error /= len(comparison)

                    # Print out errors
                    print("")
                    print("testing data points: ", len(test_set))
                    print("training data points: ", len(training_set))
                    print("--------------------------------")
                    print("RESULTS FOR L=%s MODULI (lambda=%f)"%(str_rank,reg[i]))
                    print("-----------------------------------------------------")
                    print("STD", np.sqrt(intrins_dev))
                    print("ABS RMSE", np.sqrt(abs_error))
                    print("RMSE = %.4f %%"%(100. * np.sqrt(np.abs(abs_error / intrins_dev))))

                # Finally, get predictions for the entire cartesian tensor
                ns = len(prediction)
                predcart = regression_utils.convert_spherical_to_cartesian(pred,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)
                testcart = regression_utils.convert_spherical_to_cartesian(pte ,degen,ns,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list)

                # Print out predictions
                if peratom:
                    corrfile = open("prediction_cartesian.txt","w")
                    for i in range(ns):
                        print(' '.join(str(e) for e in list(np.split(testcart,ns)[i]*nattest[i])),"  ",' '.join(str(e) for e in list(np.split(predcart,ns)[i]*nattest[i])),"  ",str(nattest[i]), file=corrfile)
                    corrfile.close()
                else:
                    corrfile = open("prediction_cartesian.txt","w")
                    for i in range(ns):
                        print(' '.join(str(e) for e in list(np.split(testcart,ns)[i])),"  ",' '.join(str(e) for e in list(np.split(predcart,ns)[i])), file=corrfile)
                    corrfile.close()

    else:
        # Do spherical regression, without environmental sparsification
        if (sparsify == None):
    
            # Read-in kernels
            print("Loading kernel matrices...")
    
            kr = np.load(kernels)
            kernel = kr
    
            print("...Kernels loaded.")
    
            # Put tensor into float form
            spherical_tensor = np.array([i.split() for i in tens]).astype(float)
    
            int_rank = int(rank[-1])
    
            sagpr_utils.do_sagpr_spherical(kernel,spherical_tensor,reg,rank_str=str(rank),nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=prediction,get_meantrain=True,mode=mode,wfile=weights,fnames=[args.features,kernels],jitter=jitter[0])
    
        # Do spherical regression, with environmental sparsification
        else:

            if (len(sparsify) != 2):
                print("ERROR: two kernels must be specified!")
                sys.exit(0)
    
            # We want to sparsify on the rows of the kernel, so we're going to load in the sparsification kernels
            print("Loading sparsification kernels...")
    
            kr1 = np.load(sparsify[0])
            kr2 = np.load(sparsify[1])
            nN = len(kr1)
            nM = len(kr1[0])
            if (len(kr2) != nM or len(kr2[0]) != nM):
                print("ERROR: kernel matrices have incorrect dimensions!")
                sys.exit(0)
            kernel = [kr1,kr2]
    
            print("...Kernels loaded.")
    
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
            sagpr_utils.do_sagpr_spherical(sparse_kernel,spherical_tensor,reg,rank_str=str(rank),nat=nat,fractrain=fractrain,rdm=rdm,sel=sel,peratom=peratom,prediction=False,reg_matr=reg_matr,get_meantrain=False,mode=mode,wfile=weights,fnames=[args.features,sparsify],jitter=jitter[0])
    
            # Deal with meantrain: put this into the weights so that we can do prediction
            if (int_rank == 0):
                wts = np.load(weights + "_" + str(rank) + ".npy",allow_pickle=True)
                wts[5] = meantrain
                np.save(weights + "_" + str(rank) + ".npy",wts)

            # If we have asked for prediction here, get predictions for the testing set we created
            if (prediction):

                # Multiply weights by kernel to get predictions, reshaping kernel if necessary
                if (int_rank>0):
                    shp = np.shape(kte)
                    kte = kte.transpose(0,2,1,3).reshape(shp[0]*shp[2],shp[1]*shp[3])
                    wts = np.load(weights + "_" + str(rank) + ".npy",allow_pickle=True)
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

                # Print out errors
                print("")
                print("testing data points: ", len(test_set))
                print("training data points: ", len(training_set))
                print("--------------------------------")
                print("RESULTS FOR L=%i MODULI (lambda=%f)"%(int_rank,reg))
                print("-----------------------------------------------------")
                print("STD", np.sqrt(intrins_dev))
                print("ABS RMSE", np.sqrt(abs_error))
                print("RMSE = %.4f %%"%(100. * np.sqrt(np.abs(abs_error / intrins_dev))))

if __name__=="__main__":
    main()
