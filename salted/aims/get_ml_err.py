import numpy as np
import time
import sys
import inp

def main():
    from salted.sys_utils import read_system
    spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    start_time = time.time()

    dirname = inp.path2qm+inp.predict_data
    av_err = 0
    errs = np.zeros(ndata)
    g = open('ml_maes','w+')
    for i in range(1,ndata+1):
        dirn = dirname+str(i)+'/'
#        f = open(dirn+'rho_scf.out')
#        r_con = [float(line.split()[-1]) for line in f]
        r_con = np.loadtxt(dirn+'rho_scf.out')
        r_con.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
#        f = open(dirn+'rho_df.out')
#        r_ri = [float(line.split()[-1]) for line in f]
        r_ri = np.loadtxt(dirn+'rho_ml.out')
        r_ri.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
#        f = open(dirn+'partition_tab.out')
#        part = [float(line.split()[-1]) for line in f]
        part = np.loadtxt(dirn+'partition_tab.out')
        part.view('f8,f8,f8,f8').sort(order=['f0','f1','f2'],axis = 0)
        err = np.abs(r_ri[:,3]-r_con[:,3])
        norm = np.dot(r_con[:,3],part[:,3])
        int_err = np.dot(err,part[:,3])*100/norm
        errs[i-1] = int_err
        g.write(str(i)+'    '+str(int_err)+'\n')
        g.flush()

    g.close()
    av_err = np.average(errs)
    sem = np.std(errs)/np.sqrt(ndata)

    print('% MAE =', av_err)
#    print(round(time.time() - start_time,1),'seconds')

main()
