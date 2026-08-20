import os
import argparse
import sys
import os.path as osp

import numpy as np

from salted.sys_utils import ParseConfig, read_system
inp = ParseConfig().parse_input()

def add_command_line_arguments_contraction():
    parser = argparse.ArgumentParser()
    parser.add_argument("-vl", "--validation", action='store_true', help="Move SALTED-predicted coefficients for the validations into the relevant AIMS data folders")
    args = parser.parse_args()
    return args

args = add_command_line_arguments_contraction()
validation = args.validation
ntrain = int(inp.gpr.trainfrac*inp.gpr.Ntrain)

if validation:
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()
    dn = os.path.join(inp.qm.path2qm, 'data')
    # define validation set
    rdir = f"regrdir_{inp.salted.saltedname}"
    trainrangetot = np.loadtxt(osp.join(
        inp.salted.saltedpath, rdir, f"training_set_N{inp.gpr.Ntrain}.txt"
    ), int)
    testset = np.setdiff1d(list(range(ndata)),trainrangetot)
else:
    dn = os.path.join(inp.qm.path2qm, inp.prediction.predict_data)
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system(filename=inp.prediction.filename,spelist = inp.system.species, dfbasis = inp.qm.dfbasis)
    testset = list(range(ndata))

testset = [x+1 for x in testset]

es = []
xcs = []
eles = []
n_atoms = []


for k,i in enumerate(testset):
    e = []
    xc = []
    har = []
    ele = []
    dirn = os.path.join(dn, str(i))

    f1 = open(os.path.join(dirn, 'aims.out'))
    for line in f1:
        if line.find('| Number of atoms') != -1:
           n_atoms.append(line.split()[5])
        elif line.find('| Electrostatic energy') != -1:
            ele.append(line.split()[6])
        elif line.find('XC energy correction') != -1:
            xc.append(line.split()[7])
        elif line.find('| Electronic free energy per atom') != -1:
            e.append(line.split()[7])
        else:
            continue

    f1 = open(os.path.join(dirn, 'aims_predict.out'))
    for line in f1:
        if line.find('| Electrostatic energy') != -1:
            ele.append(line.split()[6])
        elif line.find('XC energy correction') != -1:
            xc.append(line.split()[7])
        elif line.find('| Electronic free energy per atom') != -1:
            e.append(line.split()[7])
        else:
            continue

    es.append([])
    xcs.append([])
    eles.append([])
    es[k].append(e[-2])
    xcs[k].append(xc[-2])
    eles[k].append(ele[-2])
    es[k].append(e[-1])
    xcs[k].append(xc[-1])
    eles[k].append(ele[-1])
    
es = np.array(es,dtype = float)
xcs = np.array(xcs,dtype = float)
eles = np.array(eles,dtype = float)
n_atoms = np.array(n_atoms,dtype = float)

for i in range(2):
    xcs[:,i] /= n_atoms
    eles[:,i] /= n_atoms

if validation:
    fname_prefix = "validation_reference"
else:
    fname_prefix = "predict_reference"

np.savetxt(f'{fname_prefix}_electrostatic_energy.dat',np.vstack([eles[:,1],eles[:,0]]).T)
np.savetxt(f'{fname_prefix}_xc_energy.dat',np.vstack([xcs[:,1],xcs[:,0]]).T)
np.savetxt(f'{fname_prefix}_total_energy.dat',np.vstack([es[:,1],es[:,0]]).T)

print('Mean absolute errors (eV/atom):')
print('Electrostatic energy:',np.average(np.abs(eles[:,1]-eles[:,0])))
print('XC energy:',np.average(np.abs(xcs[:,1]-xcs[:,0])))
print('Total energy:',np.average(np.abs(es[:,1]-es[:,0])))
