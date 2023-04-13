import os
from sys_utils import read_system
from math import ceil,floor
from psutil import cpu_count
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

if predict:
    dirpath = os.path.join(inp.path2ml, inp.predict_soapdir)
    fname = inp.predict_filename
else:
    dirpath = os.path.join(inp.path2ml, inp.soapdir)
    fname = inp.filename
spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(filename=fname)

spe = ' '.join(inp.species)
if periodic:
    per = '-p'
else:
    per = ''
if vf < 1.0:
    svf = '-vf '+str(vf)
else:
    svf = ''

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
        cmd = ['get_power_spectrum.py','-f',fname,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-nc',str(nc),'-ns',str(ns),'-sm', 'random', '-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
        subprocess.call(cmd)

if parallel > 1:
    import ase.io
    import numpy as np
    coords = ase.io.read(fname,":")
    npoints = len(coords)
    block = ceil(npoints/parallel)
    cpt = floor(cpu_count(logical=False)/parallel)
    output = [None]*parallel*(llmax+1)

    # Split coords file into parallel blocks and calculate the SOAP descriptors for each
    for i in range(parallel):
        fname1 = str(i)+'_'+fname
        if i < parallel-1:
           ase.io.write(fname1,coords[i*block:(i+1)*block])
        else:
           ase.io.write(fname1,coords[i*block:])
    j = 0
    for l in range(llmax+1):
        for i in range(parallel):
            fname1 = str(i)+'_'+fname
            if nc > 0:
                cmd = ['srun','--exclusive','-n','1','-c',str(cpt),'get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
            else:
                cmd = ['srun','--exclusive','-n','1','get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
            output[j] = subprocess.Popen(cmd)
            j += 1

    # Wait for all blocks to finish
    for i in range(parallel*(llmax+1)):
        output[i].wait()

    # Clean Up
    for i in range(parallel):
        fname1 = str(i)+'_'+fname
        os.remove(fname1)
    for l in range(llmax+1):
        for i in range(parallel):
            block = np.load(dirpath+str(i)+'FEAT-'+str(l)+'.npy')
            natoms = np.load(dirpath+str(i)+'FEAT-'+str(l)+'_natoms.npy')
            if i == 0:
                full = block
                full_natoms = natoms
            else:
                full = np.concatenate([full,block])
                full_natoms = np.concatenate([full_natoms,natoms])
            os.remove(dirpath+str(i)+'FEAT-'+str(l)+'.npy')
            os.remove(dirpath+str(i)+'FEAT-'+str(l)+'_natoms.npy')
        np.save(dirpath+'FEAT-'+str(l)+'.npy',full)
        np.save(dirpath+'FEAT-'+str(l)+'_natoms.npy',full_natoms)


else:
    if nc > 0:
        cmd = ['get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
    else:
        cmd = ['get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
    subprocess.call(cmd)
