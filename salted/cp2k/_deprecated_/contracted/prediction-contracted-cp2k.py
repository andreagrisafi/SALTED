import os
import sys
import numpy as np
import time
import ase
from ase import io
from ase.io import read
from random import shuffle
from scipy import special
import random

from sympy.parsing import mathematica
from sympy import symbols
from sympy import lambdify

import basis

sys.path.insert(0, './')
import inp

bohr2angs = 0.529177249

spelist = inp.species
# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

kdir = inp.kerndirtest
pdir = inp.preddir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load(inp.path2ref+"averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.path2ml, pdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

ntrain = inp.Ntrain

#dirpath = os.path.join(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(reg))
dirpath = os.path.join(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# load regression weights
weights = np.load(inp.path2mlref+inp.regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")
#weights = np.load(inp.path2mlref+inp.regrdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(reg)+".npy")

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
print("Reading auxiliary basis info...")
alphas = {}
sigmas = {}
for spe in spelist:
    avals = np.loadtxt("alphas-"+str(spe)+".txt")
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            alphas[(spe,l,n)] = avals[n]
            sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

projector = {}
ncut = {}
normfact = {}
for spe in spelist:
    for l in range(lmax[spe]+1):
        projector[(spe,l)] = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector[(spe,l)].shape[-1]
        normfact[(spe,l)] = np.zeros(nmax[(spe,l)])
        for n in range(nmax[(spe,l)]):
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            normfact[(spe,l)][n] = np.sqrt(inner)

norm_projector = {}
norm_primitive = {}
nfact = {}
for spe in spelist:
    for l in range(lmax[spe]+1):
        prefac = 2.0**l*(2.0/np.pi)**0.75
        expalpha = 0.25*float(2*l + 3)
        norm_projector[(spe,l)] = np.zeros((nmax[(spe,l)],ncut[(spe,l)]))
        norm_primitive[(spe,l)] = np.zeros(nmax[(spe,l)])
        for n in range(nmax[(spe,l)]):
            norm_primitive[(spe,l)][n] = 1.0/(prefac*alphas[(spe,l,n)]**expalpha)
            norm_projector[(spe,l)][n] = projector[(spe,l)][n] * prefac*alphas[(spe,l,n)]**expalpha
        nfact[(spe,l)] = np.zeros(ncut[(spe,l)])
        for n in range(ncut[(spe,l)]):
            nfact_temp = 0.0
            for ipgf1 in range(nmax[(spe,l)]):
                for ipgf2 in range(nmax[(spe,l)]):
                    nfact_temp += norm_projector[(spe,l)][ipgf1,n] * norm_projector[(spe,l)][ipgf2,n] * 0.5 * special.gamma(l+1.5) / ( (alphas[(spe,l,ipgf1)] + alphas[(spe,l,ipgf2)])**(l+1.5) )
            nfact[(spe,l)][n] = np.sqrt(nfact_temp)

# compute integrals of basis functions (needed to a posteriori correction of the charge)
charge_integrals = {}
dipole_integrals = {}
for spe in spelist:
    for l in range(lmax[spe]+1):
        # charge contributions
        charge_radint = np.zeros(nmax[(spe,l)])
        dipole_radint = np.zeros(nmax[(spe,l)])
        for n in range(nmax[(spe,l)]):
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            charge_radint[n] = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )         / np.sqrt(inner)
            dipole_radint[n] = 2**float(1.0+float(l)/2.0) * sigmas[(spe,l,n)]**(4+l) * special.gamma(2.0+float(l)/2.0) / np.sqrt(inner)
        charge_radint_contr = np.dot(projector[(spe,l)].T,charge_radint)
        dipole_radint_contr = np.dot(projector[(spe,l)].T,dipole_radint)
        for n in range(ncut[(spe,l)]):
            charge_integrals[(spe,l,n)] = charge_radint_contr[n] * np.sqrt(4.0*np.pi)
            dipole_integrals[(spe,l,n)] = dipole_radint_contr[n] * np.sqrt(4.0*np.pi/3.0) 

# If total charge density is asked, read in the GTH pseudo-charge and return a radial numpy function
if inp.totcharge:
    pseudof = open(inp.pseudochargefile,"r")
    pseudochargedensity = mathematica.mathematica(pseudof.readlines()[0],{'Erf[x]':'erf(x)'})
    pseudof.close()
    rpseudo = symbols('r')
    pseudochargedensity = lambdify(rpseudo, pseudochargedensity, modules=['numpy'])
    pseudochargedensity = np.vectorize(pseudochargedensity)
    nn = 100000
    dr = 5.0/nn
    pseudocharge = 0.0
    for ir in range(1,nn):
        r = ir*dr
        pseudocharge += r**2*pseudochargedensity(r)
    pseudocharge *= 4*np.pi*dr
    print("Integrated pseudo-charge =", pseudocharge)


# compute error over test set
error_density = 0
variance = 0
iconf = 0
for iconf in range(ndata):

    geom = xyzfile[iconf]
    geom.wrap()
    coords = geom.get_positions()/bohr2angs

    Tsize = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                Tsize += 2*l+1

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in spelist:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                #rc = 6.0#orcuts[iii]
                #psi_nm = np.load(inp.path2ml+kdir[rc]+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                psi_nm = np.load(inp.path2ml+kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1
    
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1] 
                if l==0:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2*l+1
        ispe[spe] += 1

    # add the average spherical coefficients to the predictions 
    np.savetxt("weights.dat",weights)
    pred_coefs += Av_coeffs

    # compute integral of predicted density
    iaux = 0
    rho_int = 0.0
    nele = 0.0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        nele += pseudocharge 
        # add nuclear contribution 
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                for im in range(2*l+1):
                    # compute charge
                    if l==0 and im==0:
                        rho_int += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                    iaux += 1

    if inp.qscale:
        # rescale spherical coefficients to conserve the electronic charge
        iaux = 0
        charge = 0.0
        dipole = 0.0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            dipole += pseudocharge * coords[iat,2] 
            for l in range(lmax[spe]+1):
                for n in range(ncut[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0 and im==0:
                            pred_coefs[iaux] *= nele/rho_int
                            charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)] 
                            dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                        if l==1 and im==1:
                            # subtract electronic contribution 
                            dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)] 
                        iaux += 1
    else:
        charge = rho_int.copy()
        dipole = 0.0

    print(iconf+1, ":", "unnormalized integral =", rho_int, "rho integral =", charge, "dipole =", dipole, flush=True)
    #np.save(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(reg)+"/prediction_conf"+str(iconf)+".npy",pred_coefs)
    np.save(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/prediction_conf"+str(iconf)+".npy",pred_coefs)

    # rescale for different CP2K normalization
    iaux = 0
    pred_coefs_renorm = np.zeros(Tsize)
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        for l in range(lmax[spe]+1):
             blocksize = ncut[(spe,l)]*(2*l+1)
             coefs_primit = np.einsum('a,ab->ab',norm_primitive[(spe,l)]/normfact[(spe,l)],np.dot(projector[(spe,l)],pred_coefs[iaux:iaux+blocksize].reshape(ncut[(spe,l)],2*l+1)))
             pred_coefs_renorm[iaux:iaux+blocksize]  = np.einsum('a,ab->ab',nfact[(spe,l)],np.dot(projector[(spe,l)].T,coefs_primit)).reshape(blocksize)
             iaux += blocksize

    # save predicted coefficients
    #np.savetxt(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(reg)+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs_renorm)
    np.savetxt(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs_renorm)
