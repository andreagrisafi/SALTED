import sys
import numpy as np
import scipy
from scipy import special
import time
import ase
from ase import io
from ase.io import read

sys.path.insert(0, './')
import inp

sys.path.insert(0, '../../src/')
import basis

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

nenv = {}
for ispe in xrange(nspecies):
    spe = spe_dict[ispe]
    nenv[spe] = 0
    for iconf in xrange(ndata):
        nenv[spe] += atom_counting[iconf,ispe]

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.zeros(nmax[(spe,0)],float)

print "computing averages..."
for iconf in xrange(ndata):
    atoms = atomic_symbols[iconf]
    #==================================================
    Proj = np.load("projections/projections_conf"+str(iconf)+".npy")
    Over = np.load("overlaps/overlap_conf"+str(iconf)+".npy")
    Coef = np.linalg.solve(Over,Proj)
    #==================================================
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat] 
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       av_coefs[spe][n] += Coef[i]
                    i += 1

print "saving averages..."
for spe in spelist:
    av_coefs[spe] /= nenv[spe]
    np.save("averages_"+str(spe)+".npy",av_coefs[spe])

print "computing baselined projections..."
for iconf in xrange(ndata):
    atoms = atomic_symbols[iconf]
    #==================================================
    totsize = 0
    for iat in xrange(natoms[iconf]):
        for l in xrange(lmax[atoms[iat]]+1):
            totsize += nmax[(atoms[iat],l)]*(2*l+1)
    #==================================================
    Proj = np.load("projections/projections_conf"+str(iconf)+".npy")
    Over = np.load("overlaps/overlap_conf"+str(iconf)+".npy")
    #==================================================
    Av_coeffs = np.zeros(totsize,float)
    i = 0
    for iat in xrange(natoms[iconf]):
        spe = atoms[iat]
        for l in xrange(lmax[spe]+1):
            for n in xrange(nmax[(spe,l)]):
                for im in xrange(2*l+1):
                    if l==0:
                       Av_coeffs[i] = av_coefs[spe][n] 
                    i += 1
    #==================================================
    Proj -= np.dot(Over,Av_coeffs)
    np.savetxt("projections/projections_conf"+str(iconf)+".dat",Proj, fmt='%.10e')
    np.savetxt("overlaps/overlap_conf"+str(iconf)+".dat", np.concatenate(Over), fmt='%.10e')
