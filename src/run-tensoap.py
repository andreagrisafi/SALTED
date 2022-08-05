import os
from sys_utils import read_system
import inp
import subprocess
import argparse

def add_command_line_arguments_contraction(parsetext):
    parser = argparse.ArgumentParser(description=parsetext)
    parser.add_argument("-ns", "--ns", type=int, default=100, help="Number of sparse features")
    parser.add_argument("-nc", "--nc", type=int, default=1000, help="Number of structures the sparse features are selected from")
    parser.add_argument("-p", "--periodic", action='store_true', help="Number of structures the sparse features are selected from")
    parser.add_argument("-vf", "--vol_frac",       type=float, default=1.0,                              help="Specify the occupied fraction of the cell")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction("")
ns = args.ns
nc = args.nc
periodic = args.periodic
vf = args.vol_frac

spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
dirpath = os.path.join(inp.path2ml, inp.soapdir)
os.environ['lmax'] = str(llmax)
os.environ['TENSOAP_FILE_IN'] = inp.filename
os.environ['TENSOAP_OUTDIR'] = dirpath
os.environ['TENSOAP_SPECIES'] = ' '.join(inp.species)
os.environ['TENSOAP_NC'] = str(nc)
os.environ['TENSOAP_NS'] = str(ns)
if periodic:
    os.environ['TENSOAP_P'] = '-p'
else:
    os.environ['TENSOAP_P'] = ''
if vf < 1.0:
    os.environ['TENSOAP_VF'] = '-vf '+str(vf)
else:
    os.environ['TENSOAP_VF'] = ''

spath = os.environ.get('SALTEDPATH')

# make directories if not exisiting
if not os.path.exists(inp.path2ml):
    os.mkdir(inp.path2ml)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

subprocess.call(['bash',spath+'/tensoap.sh'])
