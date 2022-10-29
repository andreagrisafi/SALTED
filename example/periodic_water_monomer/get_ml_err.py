import numpy as np
import time
import sys
import inp

def main():
    
    start_time = time.time()

    dirname = inp.path2qm+inp.predict_data
    av_err = 0
    n = 10
    errs = np.zeros(n)
    g = open('ml_maes','w+')
    for i in range(1,n+1):
        dirn = dirname+str(i)+'/'
        f = open(dirn+'rho_scf.out')
        r_con = [float(line.split()[-1]) for line in f]
        f = open(dirn+'rho_ml.out')
        r_ri = [float(line.split()[-1]) for line in f]
        f = open(dirn+'partition_tab.out')
        part = [float(line.split()[-1]) for line in f]
        err = np.abs(np.asarray(r_ri)-np.asarray(r_con))
        norm = np.dot(r_con,part)
        int_err = np.dot(err,part)*100/norm
        errs[i-1] = int_err
        g.write(str(i)+'    '+str(int_err)+'\n')

    g.close()
    av_err = np.average(errs)
    sem = np.std(errs)/np.sqrt(n)

    print(av_err,sem)
    print(round(time.time() - start_time,1),'seconds')

main()
