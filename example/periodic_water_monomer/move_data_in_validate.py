#from os import path
import numpy as np
import sys
import inp

def main():
        
        M = inp.Menv
        ntrain = int(inp.trainfrac*inp.Ntrain)
	i = int(sys.argv[1])-1
        t = np.load(inp.path2qm+inp.preddir+"M"+str(M)+"_eigcut"+str(int(np.log10(inp.eigcut)))+"/N_"+str(ntrain)+"/prediction_conf"+str(i)+".npy")
	n = len(t)

        dirpath = inp.path2qm+'data/'+str(i+1)
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

main()
