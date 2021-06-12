#!/usr/bin/python

import numpy as np
import sys
from random import shuffle
import scipy.linalg
from sympy.physics.quantum.cg import CG

###############################################################################################################################

def shuffle_data(ndata,sel,rdm,fractrain):
    # Populate arrays for training and testing set

    if rdm == 0:
        if (sel[0] != 'file'):
            if (sel[1] == -1):
                sel[1] = ndata
            trrangemax = np.asarray(list(range(sel[0],sel[1])),int)
        else:
            trrangemax = np.asarray(sel[1],int)
    else:
        data_list = list(range(ndata))
        shuffle(data_list)
        trrangemax = np.asarray(data_list[:rdm],int).copy()
    terange = np.setdiff1d(list(range(ndata)),trrangemax)

    ns = len(terange)
    ntmax = len(trrangemax)
    nt = int(fractrain*ntmax)
    trrange = trrangemax[0:nt]

    return [ns,nt,ntmax,trrange,terange]

###############################################################################################################################

def build_training_kernel(nt,size,ktr,reg,reg_matr):
    # Build training kernel
    if size>1:
        ktrainpred = ktr.transpose(0,2,1,3).reshape(-1,nt*size)
        if (len(reg_matr) == 0):
            # Do standard regularization
            reg_matr = np.eye(size*nt)
        ktrain = ktrainpred + reg*reg_matr# + 1e-8* np.eye(len(reg_matr))
    else:
        if (len(reg_matr) == 0):
            # Do standard regularization
            reg_matr = np.identity(nt)
        ktrain = np.real(ktr) + reg*reg_matr# + 1e-8* np.eye(len(reg_matr))
        ktrainpred = np.real(ktr)
    return [ktrain,ktrainpred]

###############################################################################################################################

def build_testing_kernel(ns,nt,size,kte):
    # Build testing kernel
    if size>1:
        ktest = kte.transpose(0,2,1,3).reshape(-1,nt*size)
    else:
        ktest = np.real(kte)
    return ktest

###############################################################################################################################

def get_spherical_components(tens,CS,threshold,keep_cols,all_sym):
    # Extract the complex spherical components of the tensors, and keep only those that are nonzero (within some threshold)


    if (len(keep_cols[-1]) > 0):
        degen = [1 + 2*keep_cols[-1][i][-1] for i in range(len(keep_cols[-1]))]
    else:
        degen = [1]
        return [np.array(tens).astype(float),[1],None,[1]]
    cumulative_degen = [sum(degen[:i]) for i in range(1,len(degen)+1)]

    # Get CR matrix
    CR = complex_to_real_transformation(degen)

    all_tens_sphr = [ [] for i in range(len(degen))]
    for i in range(len(tens)):
        # Split into spherical components
        tens_sphr = np.split(np.dot(np.array(tens[i]).astype(float),CS),cumulative_degen)
        for j in range(len(degen)):
            vtensor_out = []
            spherical = tens_sphr[j]
            # Is this a real or imaginary spherical tensor?
            if (not all_sym[-1][j]):
                spherical /= 1.0j
            # Convert to real spherical harmonic
            if (degen[j] == 1):
                vtensor_out.append(np.real(spherical[0]))
            else:
                vtensor_out.append(np.real(np.dot(CR[j],spherical)))
            all_tens_sphr[j].append(vtensor_out)

    for i in range(len(degen)):
        if (degen[i] > 1):
            all_tens_sphr[i] = np.reshape(all_tens_sphr[i],np.size(all_tens_sphr[i]))

    # Decide which ones to keep
    spherical_components = []
    keep_list            = []
    out_degen            = []
    out_CR               = []
    keep_indices         = []
    keep_sym             = []
    for i in range(len(degen)):
        if (np.linalg.norm(all_tens_sphr[i][:]) > threshold):
            spherical_components.append(all_tens_sphr[i])
            keep_list.append(keep_cols[-1][i])
            out_degen.append(degen[i])
            out_CR.append(CR[i])
            keep_sym.append(all_sym[-1][i])
            keep_indices.append(i)

    # Check to see if any of the components with the same degeneracy are linearly independent (as happens, e.g., for L=3 symmetric tensors)
    lin_dep_list  = []
    lin_dep_local = []
    for i in range(len(keep_indices)):
        for j in range(i+1,len(keep_indices)):
            if (out_degen[i] == out_degen[j]):
                if (np.linalg.matrix_rank(np.column_stack((spherical_components[i],spherical_components[j]))) <= 1):
                    avg_list = []
                    for k in range(len(spherical_components[i])):
                        if (out_degen[i] > 1):
                            if (abs(spherical_components[i][k]) > threshold):
                                avg_list.append(spherical_components[j][k] / spherical_components[i][k])
                        else:
                            if (abs(spherical_components[i][k][0]) > threshold):
                                avg_list.append(spherical_components[j][k][0] / spherical_components[i][k][0])
                    scale_factor = np.mean(avg_list)
                    lin_dep_list.append([keep_cols[-1].index(keep_list[i]),keep_cols[-1].index(keep_list[j]),scale_factor])
                    lin_dep_local.append(j)

    # If any are linearly dependent, we only keep one of them
    final_components = []
    final_keep       = []
    final_degen      = []
    final_CR         = []
    final_indices    = []
    final_sym        = []
    for i in range(len(out_degen)):
        if (not i in lin_dep_local):
            final_components.append(spherical_components[i])
            final_keep.append(keep_list[i])
            final_degen.append(out_degen[i])
            final_CR.append(out_CR[i])
            final_sym.append(keep_sym[i])
            final_indices.append(keep_indices[i])

    return [final_components,final_keep,final_CR,final_degen,lin_dep_list,final_sym]


###############################################################################################################################

def complex_to_real_transformation(sizes):
    # Transformation matrix from complex to real spherical harmonics

    matrices = []
    for i in range(len(sizes)):
        lval = int((sizes[i]-1)/2)
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range(lval):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[lval][lval] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

###############################################################################################################################

def partition_kernels_properties_spherical(data,kernel,trrange,terange,nat):
    # Partition kernels and properties for training and testing

    vtrain    = np.array([data[i] for i in trrange])
    vtest     = np.array([data[i] for i in terange])
    nattrain  = [nat[i] for i in trrange]
    nattest   = [nat[i] for i in terange]
    kttr      = np.asarray([[kernel[i,j] for j in trrange] for i in trrange],float)
    ktte      = np.asarray([[kernel[i,j] for j in trrange] for i in terange],float)

    return [vtrain,vtest,kttr,ktte,nattrain,nattest]

###############################################################################################################################

def convert_spherical_to_cartesian(outvec,degen,size,CR,CS,keep_cols,keep_list,lin_dep_list,sym_list):
    # Convert the spherical tensor representation back to Cartesian by multiplication with transformation matrices

    full_degen = [2*keep_cols[-1][i][-1] + 1 for i in range(len(keep_cols[-1]))]

    full_outvec = [np.zeros((size,full_degen[i]),dtype=complex) for i in range(len(full_degen))]

    # Fill the full_outvec array where we have values; anywhere else we leave it at zero. Do the transformation back to spherical harmonics here
    for i in range(len(keep_list)):
        scalfac = 1.0
        if (not sym_list[i]):
            scalfac = 1.0j
        fill_col = keep_cols[-1].index(keep_list[i])
        if (degen[i] == 1):
            full_outvec[fill_col] = outvec[i] * scalfac
        else:
            ov = np.split(outvec[i],len(outvec[i]) / degen[i])
            for j in range(len(ov)):
                full_outvec[fill_col][j] = np.dot(np.conj(CR[i]).T,ov[j]) * scalfac

    # Check the list of linear dependency, and include these dependencies if present
    for i in range(len(lin_dep_list)):
        full_outvec[lin_dep_list[i][1]] = full_outvec[lin_dep_list[i][0]] * lin_dep_list[i][2]

    # Now concatenate these outputs to give one array
    concat_vec = np.zeros((int(len(outvec[0])/degen[0]),sum(full_degen)),dtype=complex)
    for i in range(int(len(outvec[0])/degen[0])):
        full_outvecs = []
        for j in range(len(full_degen)):
            if (full_degen[j]==1):
                full_outvecs.append(full_outvec[j][i])
            else:
                for k in range(len(full_outvec[j][i])):
                    full_outvecs.append(full_outvec[j][i][k])
        concat_vec[i] = full_outvecs

    # Transform array back to Cartesian tensor
    cartesian = np.real(np.dot(concat_vec,np.conj(CS).T))

    return np.concatenate(cartesian)

###############################################################################################################################

def get_cartesian_to_spherical(rank):

    all_CS    = []
    all_rows  = []
    keep_cols = [ [] ]
    all_CS.append(np.array([1.0]))
    all_rows.append([[1],[1]])
    if (rank>0):
        keep_cols.append([[1]])
        all_CS.append(np.array([[1.0,0.0,-1.0],[-1.0j,0.0,-1.0j],[0.0,np.sqrt(2.0),0.0]],dtype = complex) / np.sqrt(2.0))
        all_rows.append([[0,1,2],[ [1,-1],[1,0],[1,1] ]])

    if rank > 1:
        for i in range(2,rank+1):
            new_cs = np.zeros((3**i,3**i),dtype=complex)
            # Get list of columns
            col = [[1]]
            for rn in range(2,i+1):
                col = [col[k] + [j] for k in range(len(col)) for j in range(rn+1)]
            keep_col = []
            for cc in range(len(col)):
                # Only keep the entries where no entry differs by more than 1 from any of its neighbours
                col_list = np.array(col[cc]).astype(int)
                dif_list = np.abs([col_list[n] - col_list[n-1] for n in range(1,len(col_list))])
                sum_list = [col_list[n] + col_list[n-1] for n in range(1,len(col_list))]
                if ((min(sum_list) > 0) and (max(dif_list) <= 1)):
                    keep_col.append(col[cc])
            # Now include m values
            keep_cols.append(keep_col)
            col = []
            for cc in range(len(keep_col)):
                jj = keep_col[cc][-1]
                for m in range(2*jj + 1):
                    col.append(keep_col[cc] + [m-jj])
            if (len(col) != 3**i):
                print("ERROR: the number of columns is incorrect!")
                sys.exit(0)
            # Find rows
            row = [list(map(int,np.base_repr(rr,3).rjust(i,'0'))) for rr in range(3**i)]
            for cc in range(3**i):
                for rr in range(3**i):
                    # Convert the row number into Cartesian components
                    # Use the recursion relation to find the element of this list
                    an  = row[rr][-1]
                    AA  = row[rr][:len(row[rr])-1]
                    if (len(AA)==1):
                        AA = AA[0]
                    jn  = col[cc][i-1]
                    jnm = col[cc][i-2]
                    JJ  = col[cc][:i-1]
                    mm  = col[cc][-1]
                    # Get the rows (Cartesian elements) of the previous entry of all_CS that we need for recursively building this matrix
                    new_row_index1 = AA
                    new_row1 = all_rows[-1][0].index(new_row_index1)
                    new_row_index2 = an
                    new_row2 = all_rows[1][0].index(new_row_index2)
                    for m1 in range(-jnm,jnm+1):
                        # Get the first factor, which is found from the previous entry of all_CS
                        new_col_index1 = JJ + [m1]
                        new_col1 = all_rows[-1][1].index(new_col_index1)
                        fac1 = all_CS[-1][new_row1][new_col1]
                        for m2 in range(-1,2):
                            # Get the second factor, which is found from the first entry of all_CS
                            new_col_index2 = [1,m2]
                            new_col2 = all_rows[1][1].index(new_col_index2)
                            fac2 = all_CS[1][new_row2][new_col2]
                            # Put everything together
                            new_cs[rr,cc] += fac1 * fac2 * CG(jnm,m1,1,m2,jn,mm).doit()
    
            all_CS.append(new_cs)
            all_rows.append([row,col])

    # Find the symmetries of spherical harmonics
    all_sym = []
    all_sym.append([True])
    if rank > 0:
        all_sym.append([True])
    if rank > 1:
        # We know the results for a rank-2 tensor, and we can use these to build everything else up recursively.
        all_sym.append([True,False,True])
    if rank > 2:
        for i in range(3,rank+1):
            sym = []
            for j in range(len(keep_cols[i])):
                diff = np.abs(keep_cols[i][j][-2]-keep_cols[i][j][-1])
                if (diff == 1):
                    # We take the same symmetry
                    sym.append(all_sym[-1][keep_cols[i-1].index(keep_cols[i][j][:-1])])
                elif (diff == 0):
                    # We take the opposite symmetry
                    sym.append(not all_sym[-1][keep_cols[i-1].index(keep_cols[i][j][:-1])])
                else:
                    print("ERROR: we shouldn't be here! diff = ",diff)
                    sys.exit(0)
            all_sym.append(sym)

    return [all_CS,keep_cols,all_sym]

###############################################################################################################################

def get_lvals(rank):
    # Get the lvals for a given rank

    if (rank%2 == 0):
        # Even L
        lvals = [l for l in range(0,rank+1,2)]
    else:
        # Odd L
        lvals = [l for l in range(1,rank+1,2)]

    return lvals

###############################################################################################################################

def get_degen(rank):
    # Get the degeneracies for a given rank

    lvals = get_lvals(rank)
    return [2*l+1 for l in lvals]

###############################################################################################################################
