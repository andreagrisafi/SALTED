import sys
import numpy as np
import time

def build():

    sys.path.insert(0, './')
    import inp
    
    # sparse-GPR parameters
    M = inp.Menv
    reg = inp.regul
    zeta = inp.z
    
    if inp.field:
        kdir = "kernels_"+inp.saltedname+"_field"
        fdir = "rkhs-vectors_"+inp.saltedname+"_field"
        rdir = "regrdir_"+inp.saltedname+"_field"
    else:
        kdir = "kernels_"+inp.saltedname
        fdir = "rkhs-vectors_"+inp.saltedname
        rdir = "regrdir_"+inp.saltedname
    
    # define training set size 
    ntrain = round(inp.trainfrac*inp.Ntrain)
    
    # load regression matrices
    Avec = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Avec_N"+str(ntrain)+".npy")
    totsize = Avec.shape[0]
    print("problem dimensionality:", totsize,flush=True)
    if totsize>70000:
        print("ERROR: problem dimension too large, minimize directly loss-function instead!")
        sys.exit(0)
    Bmat = np.load(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/Bmat_N"+str(ntrain)+".npy")
    
    start = time.time()
    
    w = np.linalg.solve(Bmat+np.eye(totsize)*reg,Avec)
    
    print("regression time:", (time.time()-start)/60, "minutes")
    
    np.save(inp.saltedpath+rdir+"/M"+str(M)+"_zeta"+str(zeta)+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy",w)

    return

if __name__ == "__main__":
    build()
