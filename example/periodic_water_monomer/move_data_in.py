#from os import path
import numpy as np
import sys
import inp
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-pr", "--predict", action='store_true', help="Prepare geometries for a true prediction")
    parser.add_argument("-i", "--iconf", type=int, default=0, help="Move the predicted coefficients of the iconfth structure")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
predict = args.predict
i = args.iconf

if predict:
    pdir = inp.predict_coefdir
else:
    pdir = inp.valcdir

M = inp.Menv
ntrain = int(inp.trainfrac*inp.Ntrain)
t = np.load(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(i-1)+".npy")
n = len(t)

if predict:
    dirpath = inp.path2qm+inp.predict_data+str(i)
else:
    dirpath = inp.path2qm+'data/'+str(i)
    
idx = np.loadtxt(dirpath+'/idx_prodbas.out').astype(int)
idx -= 1
idx = list(idx)
idx_rev = []
for i in range(n):
    idx_rev.append(idx.index(i))
#	np.savetxt('../idx_prodbas_rev.out',idx_rev)
#	idx_rev = np.loadtxt('../idx_prodbas_rev.out').astype(int)
cs_list = np.loadtxt(dirpath+'/prodbas_condon_shotley_list.out').astype(int)
cs_list -= 1
cs_list = list(cs_list)

t = t[idx_rev]
for j in cs_list:
    t[j] *= -1

np.savetxt(dirpath+'/ri_restart_coeffs_predicted.out',t)

