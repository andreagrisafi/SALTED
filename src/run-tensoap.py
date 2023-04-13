import os
from sys_utils import read_system
from math import ceil
import sys
sys.path.insert(0, './')
import inp
import subprocess
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-ns", "--ns", type=int, default=100, help="Number of sparse features")
    parser.add_argument("-nc", "--nc", type=int, default=1000, help="Number of structures the sparse features are selected from")
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

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
if predict:
    dirpath = os.path.join(inp.path2ml, inp.predict_soapdir)
    fname = inp.predict_filename
else:
    dirpath = os.path.join(inp.path2ml, inp.soapdir)
    fname = inp.filename

#os.environ['TENSOAP_FILE_IN'] = fname
os.environ['lmax'] = str(llmax)
os.environ['TENSOAP_OUTDIR'] = dirpath
os.environ['TENSOAP_SPECIES'] = ' '.join(inp.species)
spe = ' '.join(inp.species)
os.environ['TENSOAP_NC'] = str(nc)
os.environ['TENSOAP_NS'] = str(ns)
os.environ['TENSOAP_RC'] = str(rc)
os.environ['TENSOAP_SG'] = str(sg)
os.environ['TENSOAP_D'] = '-d '+str(dummy)
if periodic:
    os.environ['TENSOAP_P'] = '-p'
    per = '-p'
else:
    os.environ['TENSOAP_P'] = ''
    per = ''
if vf < 1.0:
    os.environ['TENSOAP_VF'] = '-vf '+str(vf)
    svf = '-vf '+str(vf)
else:
    os.environ['TENSOAP_VF'] = ''
    svf = ''

spath = os.environ.get('SALTEDPATH')

#os.environ['lmax'] = '0' #TEST
#llmax = 0 #TEST

# make directories if not exisiting
if not os.path.exists(inp.path2ml):
    os.mkdir(inp.path2ml)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

if parallel > 0:
    import ase.io
    import numpy as np
    coords = ase.io.read(fname,":")
    npoints = len(coords)
    block = ceil(npoints/parallel)
    output = [None]*parallel*(llmax+1)
    if nc > 0:
        for l in range(llmax+1):
            if os.path.exists(dirpath+'FEAT-'+str(l)+'_Amat.npy'): continue
#            output[l] = subprocess.call(['bash',spath+'/tensoap_sparsify.sh',fname,str(l)])
            cmd = ['get_power_spectrum.py','-f',fname,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-nc',str(nc),'-ns',str(ns),'-sm', 'random', '-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
#            cmd += inp.species
#            cmd.append('-c')
#            cmd += inp.species
#            cmd += ['-nc',str(nc),'-ns',str(ns),'-sm', 'random', '-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
            subprocess.call(cmd)
#            subprocess.call(['get_power_spectrum.py','-f',fname,'-lm',str(l),per,'-vf',str(vf),'-s',inp.species,'-s',inp.species,'-nc',str(nc),'-ns',str(ns),'-sm','random', '-o',dirpath+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)])
#       for l in range(llmax+1):
#           output[l].communicate()

    for i in range(parallel):
        fname1 = str(i)+'_'+fname
        if i < parallel-1:
           ase.io.write(fname1,coords[i*block:(i+1)*block])
        else:
           ase.io.write(fname1,coords[i*block:])
#        os.environ['TENSOAP_FILE_IN'] = fname1
    for l in range(llmax+1):
        for i in range(parallel):
            fname1 = str(i)+'_'+fname
            if nc > 0:
                cmd = ['srun','--exclusive','-n','1','get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-sf',dirpath+'FEAT-'+str(l),'-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
            else:
                cmd = ['srun','--exclusive','-n','1','get_power_spectrum.py','-f',fname1,'-lm',str(l),per,'-vf',str(vf),'-s']+inp.species+['-c']+inp.species+['-o',dirpath+str(i)+'FEAT-'+str(l),'-rc',str(rc),'-sg',str(sg),'-d',str(dummy)]
            output[i] = subprocess.Popen(cmd)
#       output[i] = subprocess.Popen(['srun','--exclusive','-n','1','bash',spath+'/tensoap.sh',fname1,str(i),'&'])

    # Wait for all blocks to finish
    for i in range(parallel):
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
    subprocess.call(['bash',spath+'/tensoap.sh',fname])
