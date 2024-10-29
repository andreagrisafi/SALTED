#from os import path
import numpy as np
import sys
import inp
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-pr", "--predict", action='store_true', help="Prepare geometries for a true prediction")
    parser.add_argument("-i", "--iconf", type=int, default=0, help="Move the predicted coefficients of the iconfth structure")
    parser.add_argument("-r", "--response", action='store_true', help="Move the predicted coefficients of the density response")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
predict = args.predict
iconf = args.iconf
response = args.response

if predict:
    pdir = inp.predict_coefdir
else:
    pdir = inp.valcdir

if predict:
    dirpath = inp.path2qm+inp.predict_data+str(iconf)
else:
    dirpath = inp.path2qm+'data/'+str(iconf)
    
# idx = np.loadtxt(dirpath+'/idx_prodbas.out').astype(int)
# idx -= 1
# idx = list(idx)
# idx_rev = []
# n = len(idx)
# for i in range(n):
#     idx_rev.append(idx.index(i))
# cs_list = np.loadtxt(dirpath+'/prodbas_condon_shotley_list.out').astype(int)
# cs_list -= 1
# cs_list = list(cs_list)

M = inp.Menv
ntrain = int(inp.trainfrac*inp.Ntrain)

if response:
    l = 3
    rdir = ['x/','y/','z/']
else:
    l = 1
    
for k in range(l):

    if response:
        prdir = pdir+rdir[k]
        t = np.load(inp.path2qm+prdir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(iconf-1)+".npy")
    else:
        t = np.load(inp.path2qm+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(iconf-1)+".npy")

#    t = t[idx_rev]
#    for j in cs_list:
#        t[j] *= -1

    if response:
        np.savetxt(dirpath+'/ri_rho1_restart_coeffs_predicted_'+str(k+1)+'.out',t)
    else:
        np.savetxt(dirpath+'/ri_restart_coeffs_predicted.out',t)
