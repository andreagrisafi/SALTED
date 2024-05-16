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

spelist = inp.species

nele_perat = {}
nele_perat["Au"] = 11.0

bohr2angs = 0.529177210670

# read basis
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
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

kdir = inp.kerndir
pdir = inp.preddir
rdir = inp.regrdir

coefdir = inp.coefdir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

av_coefs = {}
for spe in spelist:
    av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

dirpath = os.path.join(inp.path2ml, pdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+pdir, "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

#kdir = {}
#rcuts = [6.0]
## get truncated size
#for rc in rcuts:
#    kdir[rc] = "kernels_rc"+str(rc)+"-sg"+str(rc/10)+"/"

#orcuts = np.loadtxt("optimal_rcuts.dat")

trainrangetot = np.loadtxt("training_set_N"+str(inp.Ntrain)+".txt",int)
ntrain = int(inp.trainfrac*len(trainrangetot))
#trainrangetot = np.loadtxt("training_set_N192.txt",int)
testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

dirpath = os.path.join(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# load regression weights
weights = np.load(inp.path2ml+rdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")

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


dirpath = os.path.join(inp.path2ml,inp.resdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+inp.resdir,"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2ml+inp.resdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
# compute error over test set
dfile = open(inp.path2ml+inp.resdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/dipoles.dat","w")
qfile = open(inp.path2ml+inp.resdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/charges.dat","w")
efile = open(inp.path2ml+inp.resdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/errors.dat","w")
error_density = 0
variance = 0
for iconf in testrange:
#for iconf in range(ndata):

    geom = xyzfile[iconf]
    geom.wrap()
    coords = geom.get_positions()/bohr2angs

    # load reference
    ref_coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    overl = np.load(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy")
    ref_projs = np.dot(overl,ref_coefs)
    Tsize = len(ref_coefs)

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
    pred_coefs += Av_coeffs

    # compute integral of predicted density
    iaux = 0
    rho_int = 0.0
    ref_rho_int = 0.0
    nele = 0.0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        nele += nele_perat[spe]
        for l in range(lmax[spe]+1):
            for n in range(ncut[(spe,l)]):
                if l==0:
                    rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                    ref_rho_int += charge_integrals[(spe,l,n)] * ref_coefs[iaux]
                iaux += 2*l+1

    if inp.qscale:
        # rescale spherical coefficients to conserve the electronic charge
        iaux = 0
        charge = 0.0
        dipole = 0.0
        ref_dipole = 0.0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            dipole += pseudocharge * coords[iat,2]
            ref_dipole += pseudocharge * coords[iat,2]
            for l in range(lmax[spe]+1):
                for n in range(ncut[(spe,l)]):
                    for im in range(2*l+1):
                        if l==0 and im==0:
                            pred_coefs[iaux] *= nele/rho_int
                            ref_coefs[iaux] *= nele/ref_rho_int
                            charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                            dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                            ref_dipole -= ref_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                        if l==1 and im==1:
                            # subtract electronic contribution 
                            dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                            ref_dipole -= ref_coefs[iaux] * dipole_integrals[(spe,l,n)]
                        iaux += 1
        print(iconf+1,ref_dipole,dipole,file=dfile)
        print(iconf+1,ref_rho_int,rho_int,file=qfile)
    else:
        charge = rho_int.copy()

    # save predicted coefficients
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
    np.savetxt(inp.path2ml+pdir+"M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs_renorm)

    # compute predicted density projections <phi|rho>
    pred_projs = np.dot(overl,pred_coefs)

    # compute error
    error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    error_density += error
    ref_projs -= np.dot(overl,Av_coeffs)
    ref_coefs -= Av_coeffs
    var = np.dot(ref_coefs,ref_projs)
    variance += var
    print(iconf+1,np.sqrt(error/var)*100,file=efile)
    print(iconf+1, ":", np.sqrt(error/var)*100, "% RMSE", flush=True)
    #print(iconf+1, ":", "rho integral =", rho_int, "normalized rho integral =", charge, "ref_dipole =", ref_dipole, "dipole =",dipole, ", error =", np.sqrt(error/var)*100, "% RMSE", flush=True)

    # UNCOMMENT TO CHECK PREDICTIONS OF <phi|rho-rho_0>
    # ------------------------------------------------- 
    #pred_projs = np.dot(overl,pred_coefs-Av_coeffs)
    #av_projs = np.dot(overl,Av_coeffs)
    #iaux = 0
    #for iat in range(natoms[iconf]):
    #    spe = atomic_symbols[iconf][iat]
    #    for l in range(lmax[spe]+1):
    #        for n in range(nmax[(spe,l)]):
    #            for im in range(2*l+1):
    #                if l==5 and im==0:
    #                    print(pred_projs[iaux],ref_projs[iaux])
    #                iaux += 1

dfile.close()
print("")
print("% RMSE =", 100*np.sqrt(error_density/variance))
