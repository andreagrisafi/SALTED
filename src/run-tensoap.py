import os
from sys_utils import read_system
from math import ceil,floor
from psutil import cpu_count,Process
from time import sleep
import sys
sys.path.insert(0, './')
import inp
import subprocess
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-ns", "--ns", type=int, default=100, help="Number of structures the sparse features are selected from")
    parser.add_argument("-nc", "--nc", type=int, default=1000, help="Number of sparse features")
    parser.add_argument("-p", "--periodic", action='store_true', help="Number of structures the sparse features are selected from")
    parser.add_argument("--predict", action='store_true', help="Calculate the SOAP descript for the structure for which the density will be predicted.")
    parser.add_argument("-vf", "--vol_frac",       type=float, default=1.0,                              help="Specify the occupied fraction of the cell")
    parser.add_argument("-rc", "--rcut",       type=float, default=4.0,                              help="Specify the SOAP cutoff radius")
    parser.add_argument("-sg", "--sigma",         type=float, default=0.3,                              help="Gaussian width")
    parser.add_argument("-d", "--dummy", type=int,default=0, help="Include a dummy atom in the SOAP descriptor, at -x (1), -y (2), or -z (3)")
    parser.add_argument("--parallel", type=int,default=0, help="Trivially parallelise the calculation of TENSOAP descriptors")
    parser.add_argument("--bare", action='store_true', help="Additionally calculate bare lambda=0 descriptors")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
ns = args.ns
nc = args.nc
periodic = args.periodic
predict = args.predict
vf = args.vol_frac
rc = args.rcut
sg = args.sigma
dummy = args.dummy
parallel = args.parallel
bare = args.bare

if predict:
    dirpath = os.path.join(inp.path2ml, inp.predict_soapdir)
    fname = inp.predict_filename
else:
    dirpath = os.path.join(inp.path2ml, inp.soapdir)
    fname = inp.filename
spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename=fname)

spe = ' '.join(inp.species)

# make directories if not exisiting
if not os.path.exists(inp.path2ml):
    os.mkdir(inp.path2ml)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
if predict and nc != 0:
    os.system('cp '+inp.path2ml+inp.soapdir+'*Amat.npy '+dirpath)
    os.system('cp '+inp.path2ml+inp.soapdir+'*fps.npy '+dirpath)
    
if nc > 0:
    for l in range(llmax+1):

        # build sparsification details if they don't already exist

        if os.path.exists(dirpath+'FEAT-'+str(l)+'_Amat.npy'): continue
        cmd = ['get_power_spectrum.py','-f',fname,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-nc',str(nc),'-ns',str(ns),'-sm', 'random', '-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
        if periodic: cmd += ['-p']
        if dummy > 0: 
            if l == 0 and bare:
                cmd2 = cmd.copy()
                cmd2[cmd2.index('-o')+1] += '-bare'
                subprocess.call(cmd2)
            cmd += ['-d',str(dummy)]
        subprocess.call(cmd)

if parallel > 1:
    import ase.io
    import numpy as np
    coords = ase.io.read(fname,":")
    npoints = len(coords)
    block = floor(npoints/parallel)
    nnodes = int(os.getenv('SLURM_JOB_NUM_NODES'))
    cpus = cpu_count(logical=False)*nnodes
    cpt = floor(cpus/parallel)
    ntasks = parallel*(llmax+1+int(bare))
    output = [None]*ntasks

    def childcount():
        cp = Process()
        num = cp.children()
        return(len(num))

    # Split coords file into parallel blocks and calculate the SOAP descriptors for each
    rem = npoints - parallel*block
    start = 0
    for i in range(parallel):
        fname1 = str(i)+'_'+fname
        end = start + block
        if i < rem: end += 1
#       if i < parallel-1:
#          ase.io.write(fname1,coords[i*block:(i+1)*block])
#       else:
#          ase.io.write(fname1,coords[i*block:])
        ase.io.write(fname1,coords[start:end])
        start=end
        print(i,end-start,end)
    j = 0
    for l in range(llmax+1):
        for i in range(parallel):
            fname1 = str(i)+'_'+fname
            if nc > 0:
                cmd = ['srun','--exclusive','-N','1','-n','1','-c',str(cpt),'get_power_spectrum.py','-f',fname1,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
#                cmd = ['get_power_spectrum.py','-f',fname1,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
            else:
                cmd = ['srun','--exclusive','-N','1','-n','1','get_power_spectrum.py','-f',fname1,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
#                cmd = ['get_power_spectrum.py','-f',fname1,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
            if periodic: cmd += ['-p']
            if dummy > 0:
                if l == 0 and bare:
                    cmd2 = cmd.copy()
                    if nc > 0: cmd2[cmd2.index('-sf')+1] += '-bare'
                    cmd2[cmd2.index('-o')+1] += '-bare'
                    output[j] = subprocess.Popen(cmd2)
                    j+=1
                cmd += ['-d',str(dummy)]
            output[j] = subprocess.Popen(cmd)
            j += 1
            while childcount() == parallel:
                sleep(1)
                for k in range(j):
                    output[k].communicate()

    # Wait for all blocks to finish
    for i in range(ntasks):
        output[i].wait()

    # Clean Up
    for i in range(parallel):
        fname1 = str(i)+'_'+fname
        os.remove(fname1)
    for l in range(llmax+1):
        start = 0
        for i in range(parallel):
            block = np.load(dirpath+str(i)+'FEAT-'+str(l)+'.npy')
            natoms = np.load(dirpath+str(i)+'FEAT-'+str(l)+'_natoms.npy')
            size = block.shape[0]
            if i == 0:
#               full = block
#               full_natoms = natoms
                dim = list(block.shape)
                dim[0] = npoints
                full = np.zeros(dim)
                dim = list(natoms.shape)
                dim[0] = npoints
                full_natoms = np.zeros(dim)
#           else:
#               full = np.concatenate([full,block])
#               full_natoms = np.concatenate([full_natoms,natoms])

            full[start:start+size] = block
            full_natoms[start:start+size] = natoms
            start += size

            os.remove(dirpath+str(i)+'FEAT-'+str(l)+'.npy')
            os.remove(dirpath+str(i)+'FEAT-'+str(l)+'_natoms.npy')
        np.save(dirpath+'FEAT-'+str(l)+'.npy',full)
        np.save(dirpath+'FEAT-'+str(l)+'_natoms.npy',full_natoms)

    if dummy > 0 and bare:
        for i in range(parallel):
            block = np.load(dirpath+str(i)+'FEAT-0-bare.npy')
            if i == 0:
                full = block
            else:
                full = np.concatenate([full,block])
            os.remove(dirpath+str(i)+'FEAT-0-bare.npy')
            os.remove(dirpath+str(i)+'FEAT-0-bare_natoms.npy')
        np.save(dirpath+'FEAT-0-bare.npy',full)

else:
    for l in range(llmax+1):
        if nc > 0:
            cmd = ['get_power_spectrum.py','-f',fname,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
        else:
            cmd = ['get_power_spectrum.py','-f',fname,'-lm',str(l),'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg)]
        if periodic: cmd += ['-p']
        if dummy > 0:
            if l == 0 and bare:
                cmd2 = cmd.copy()
                if nc > 0: cmd2[cmd2.index('-sf')+1] += '-bare'
                cmd2[cmd2.index('-o')+1] += '-bare'
                subprocess.call(cmd2)
            cmd += ['-d',str(dummy)]
        subprocess.call(cmd)
