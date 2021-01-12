import sys
import numpy as np
import scipy
from scipy import special
import time
import ase
from ase import io
from ase.io import read

import basis
sys.path.insert(0, './')
import inp


# read species
spelist = inp.species
spe_dict = {}
for i in xrange(len(spelist)):
    spe_dict[i] = spelist[i] 

# read basis
[llmax,lmax,nnmax,nmax] = basis.basiset(inp.basis)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in xrange(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

#==================== species array
species = np.sort(list(set(np.array([item for sublist in atomic_valence for item in sublist]))))
nspecies = len(species)
spec_list = []
spec_list_per_conf = {}
atom_counting = np.zeros((ndata,nspecies),int)
for iconf in xrange(ndata):
    spec_list_per_conf[iconf] = []
    for iat in xrange(natoms[iconf]):
        for ispe in xrange(nspecies):
            if atomic_valence[iconf][iat] == species[ispe]:
               atom_counting[iconf,ispe] += 1
               spec_list.append(ispe)
               spec_list_per_conf[iconf].append(ispe)
spec_array = np.asarray(spec_list,int)

#trainrangetot = np.loadtxt("training_set.txt",int)
#testrange = np.setdiff1d(range(ndata),trainrangetot)
#ntest = len(testrange)

for nnn in xrange(7):
    Projs = np.zeros((ndata*4*1),float)
    #Projs = np.zeros((ntest*4*1),float)
    j1=0
    #for iconf in testrange:
    for iconf in xrange(ndata):
        atoms = atomic_symbols[iconf]
        #==================================================
        totsize = 0
        for iat in xrange(natoms[iconf]):
            for l in xrange(lmax[atoms[iat]]+1):
                totsize += nmax[(atoms[iat],l)]*(2*l+1)
        #==================================================
        Proj = np.load("projections/projections_conf"+str(iconf)+".npy")
        #Proj1 = np.load("projections/projections_conf"+str(iconf)+".npy")
        #Proj2 = np.load("projections/prediction_conf"+str(iconf)+".npy")
        #Over = np.load("overlaps_save/overlap_conf"+str(iconf)+".npy")
        #==================================================
        #Proj_new = np.zeros(totsize,float)
        #Over_new = np.zeros((totsize,totsize),float)
        i1 = 0
        #j1 = 0
        for iat in xrange(natoms[iconf]):
            spe1= atoms[iat]
            for l1 in xrange(lmax[spe1]+1):
                for n1 in xrange(nmax[(spe1,l1)]):
                    for im1 in xrange(2*l1+1):
                        if l1==0 and n1==nnn:
                            Projs[j1] = Proj[i1]
                            #Projs[0,j1] = Proj1[i1]
                            #Projs[1,j1] = Proj2[i1]
                            j1+=1
                        i1 += 1
        #                i2 = 0
        #                j2 = 0
        #                for jat in xrange(natoms[iconf]):
        #                    spe2 = atoms[jat]
        #                    for l2 in xrange(lmax[spe2]+1):
        #                        for n2 in xrange(nmax[(spe2,l2)]):
        #                            for im2 in xrange(2*l2+1):
        #                                if l1==0:
        #                                    if l2==0:
        #                                        Over_new[j1,j2] = Over[i1,i2]
        #                                        j2 += 1
        #                                i2 += 1
        #                if l1==0 and n1==0:
        #                    Proj_new[j1] = Proj[i1]
        #                    j1+=1
        #                i1 += 1
        #np.save("projections/projections_conf"+str(iconf)+".npy",Proj_new[:j1])
        #np.save("overlaps/overlap_conf"+str(iconf)+".npy",Over_new[:j1][:,:j1])
    np.save("TEST_PROJS/S-projections_"+str(nnn)+".npy",Projs)
    #np.save("S-predictions_"+str(nnn)+".npy",Projs)
