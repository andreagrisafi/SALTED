import os
import sys

import numpy as np

import inp

dn = os.path.join(inp.path2qm, inp.predict_data)
l = os.listdir(os.path.join(dn, "geoms"))
nfiles = len(l)
testset = list(range(nfiles))
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

    f1 = open(dirn+'aims.out')
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

    f1 = open(dirn+'aims_predict.out')
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

np.savetxt('predict_reference_electrostatic_energy.dat',eles[:,0])
np.savetxt('predict_reference_xc_energy.dat',xcs[:,0])
np.savetxt('predict_reference_total_energy.dat',es[:,0])
np.savetxt('prediction_electrostatic_energy.dat',eles[:,1])
np.savetxt('prediction_xc_energy.dat',xcs[:,1])
np.savetxt('prediction_total_energy.dat',es[:,1])

print('Mean absolute errors (eV/atom):')
print('Electrostatic energy:',np.average(np.abs(eles[:,1]-eles[:,0])))
print('XC energy:',np.average(np.abs(xcs[:,1]-xcs[:,0])))
print('Total energy:',np.average(np.abs(es[:,1]-es[:,0])))
