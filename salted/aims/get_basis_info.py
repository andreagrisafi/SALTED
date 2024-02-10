import os
import os.path as osp

import numpy as np
from ase.io import read

from salted import basis
import inp

# read species
spelist = inp.species

xyzfile = read(inp.filename,":")

done_list = []

dirpath = osp.join(inp.path2qm, 'data')

afname = basis.__file__

n_list = {}
l_list = {}
for iconf in range(len(xyzfile)):
    natoms = len(xyzfile[iconf])
    atomic_symbols = xyzfile[iconf].get_chemical_symbols()
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if spe not in done_list:
            f = open(osp.join(dirpath, str(iconf+1), 'basis_info.out'))
            l = 0
            while True:
                line = f.readline()
                if not line:
                    f.close()
                    break

                line = line.split()

                if line[0] == 'atom' and int(line[1])-1 == iat:
                    read_atom = True
                    continue

                if line[1] == 'atom' and int(line[2])-1 == iat:
                    read_atom = False
                    l_list[spe] = l
                    f.close()
                    break

                if read_atom:
                    n_list[spe,l] = int(line[6])
                    l += 1


            done_list.append(spe)
            if sorted(done_list) == sorted(spelist):
                f = open('new_basis_entry','w+')
                g = open(afname,'a')
                f.write('   if basis=="'+inp.dfbasis+'":\n\n')
                g.write('   if basis=="'+inp.dfbasis+'":\n\n')
                for spe in spelist:
                    f.write('      lmax["'+spe+'"] = '+str(l_list[spe]-1)+'\n')
                    g.write('      lmax["'+spe+'"] = '+str(l_list[spe]-1)+'\n')
                f.write('\n')
                g.write('\n')
                for spe in spelist:
                    for l in range(l_list[spe]):
                        f.write('      nmax[("'+spe+'",'+str(l)+')] = '+str(n_list[spe,l])+'\n')
                        g.write('      nmax[("'+spe+'",'+str(l)+')] = '+str(n_list[spe,l])+'\n')
                    f.write('\n')
                    g.write('\n')
                f.write('      return [lmax,nmax]\n\n')
                g.write('      return [lmax,nmax]\n\n')

                f.close()
                g.close()
                exit




