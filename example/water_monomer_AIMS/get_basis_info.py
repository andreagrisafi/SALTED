import inp
import numpy as np
from ase.io import read

# read species
spelist = inp.species

xyzfile = read(inp.filename,":")

done_list = []

dirpath = inp.path2qm + 'data/'

n_list = {}
l_list = {}
for iconf in range(len(xyzfile)):
    natoms = len(xyzfile[iconf])
    atomic_symbols = xyzfile[iconf].get_chemical_symbols()
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        if spe not in done_list:
            f = open(dirpath+str(iconf+1)+'/basis_info.out')
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
                f.write('   if basis=="'+inp.dfbasis+'":\n\n')
                for spe in spelist:
                    f.write('      lmax["'+spe+'"] = '+str(l_list[spe]-1)+'\n')
                f.write('\n')
                for spe in spelist:
                    for l in range(l_list[spe]):
                        f.write('      nmax[("'+spe+'",'+str(l)+')] = '+str(n_list[spe,l])+'\n')
                    f.write('\n')

                f.close()
                exit




