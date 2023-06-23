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

species = inp.species

bohr2angs = 0.529177210670

# read basis
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in species:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# number of sparse environments
M = inp.Menv
eigcut = inp.eigcut
reg = inp.regul

coefdir = inp.coefdir

# system parameters
atomic_symbols = []
natoms = np.zeros(ndata,int)
for i in range(ndata):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

dirpath = os.path.join(inp.saltedpath, "validations_"+inp.saltedname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"validations_"+inp.saltedname+"/", "M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# define test set
trainrangetot = np.loadtxt(inp.saltedpath+"regrdir_"+inp.saltedname+"/training_set_N"+str(inp.Ntrain)+".txt",int)
ntrain = int(inp.trainfrac*len(trainrangetot))
testrange = np.setdiff1d(list(range(ndata)),trainrangetot)

dirpath = os.path.join(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# load regression weights
weights = np.load(inp.saltedpath+"regrdir_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/weights_N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+".npy")

if inp.qmcode=="cp2k":
    # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
    print("Reading auxiliary basis info...")
    alphas = {}
    sigmas = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            avals = np.loadtxt(spe+"-"+inp.dfbasis+"-alphas-L"+str(l)+".dat")
            if nmax[(spe,l)]==1:
                alphas[(spe,l,0)] = float(avals)
                sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
            else:
                for n in range(nmax[(spe,l)]):
                    alphas[(spe,l,n)] = avals[n]
                    sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
    
    # compute integrals of basis functions (needed to a posteriori correction of the charge)
    charge_integrals = {}
    dipole_integrals = {}
    for spe in species:
        for l in range(lmax[spe]+1):
            charge_integrals_temp = np.zeros(nmax[(spe,l)])
            dipole_integrals_temp = np.zeros(nmax[(spe,l)])
            for n in range(nmax[(spe,l)]):
                inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                dipole_radint = 2**float(1.0+float(l)/2.0) * sigmas[(spe,l,n)]**(4+l) * special.gamma(2.0+float(l)/2.0)
                charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
                dipole_integrals[(spe,l,n)] = dipole_radint * np.sqrt(4.0*np.pi/3.0) / np.sqrt(inner)

# If total charge density is asked, read in the GTH pseudo-charge and return a radial numpy function
#if inp.totcharge:
#    pseudof = open(inp.pseudochargefile,"r")
#    pseudochargedensity = mathematica.mathematica(pseudof.readlines()[0],{'Erf[x]':'erf(x)'})
#    pseudof.close()
#    rpseudo = symbols('r')
#    pseudochargedensity = lambdify(rpseudo, pseudochargedensity, modules=['numpy'])
#    pseudochargedensity = np.vectorize(pseudochargedensity)
#    nn = 100000
#    dr = 5.0/nn
#    pseudocharge = 0.0
#    for ir in range(1,nn):
#        r = ir*dr
#        pseudocharge += r**2*pseudochargedensity(r)
#    pseudocharge *= 4*np.pi*dr
#    print("Integrated pseudo-charge =", pseudocharge)


dirpath = os.path.join(inp.saltedpath,"validations_"+inp.saltedname)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"validations_"+inp.saltedname+"/","M"+str(M)+"_eigcut"+str(int(np.log10(eigcut))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/","N"+str(ntrain)+"_reg"+str(int(np.log10(reg))))
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# Load spherical averages if required
if inp.average:
    av_coefs = {}
    for spe in inp.species:
        av_coefs[spe] = np.load("averages_"+str(spe)+".npy")

# compute error over test set
efile = open(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/errors.dat","w")
if inp.qmcode=="cp2k":
    dfile = open(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/dipoles.dat","w")
    qfile = open(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/charges.dat","w")

error_density = 0
variance = 0
for iconf in testrange:

    # load reference
    ref_coefs = np.load(inp.saltedpath+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
    overl = np.load(inp.saltedpath+"overlaps/overlap_conf"+str(iconf)+".npy")
    ref_projs = np.dot(overl,ref_coefs)
    Tsize = len(ref_coefs)

    # compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    iii = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe]+1):
            for n in range(nmax[(spe,l)]):
                psi_nm = np.load(inp.saltedpath+"kernels_"+inp.saltedname+"/spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/psi-nm_conf"+str(iconf)+".npy") 
                Mcut = psi_nm.shape[1]
                C[(spe,l,n)] = np.dot(psi_nm,weights[isize:isize+Mcut])
                isize += Mcut
                iii += 1
        
    # fill vector of predictions
    pred_coefs = np.zeros(Tsize)
    if inp.average:
        Av_coeffs = np.zeros(Tsize)
    i = 0
    for iat in range(natoms[iconf]):
        spe = atomic_symbols[iconf][iat]
        if spe in species:
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    pred_coefs[i:i+2*l+1] = C[(spe,l,n)][ispe[spe]*(2*l+1):ispe[spe]*(2*l+1)+2*l+1]
                    if inp.average and l==0:
                        Av_coeffs[i] = av_coefs[spe][n]
                    i += 2*l+1
            ispe[spe] += 1

    # add back spherical averages if required
    if inp.average:
        pred_coefs += Av_coeffs

    if inp.qmcode=="cp2k":
        
        geom = xyzfile[iconf]
        geom.wrap()
        coords = geom.get_positions()/bohr2angs
   
        if inp.average: 
            # compute integral of predicted density
            iaux = 0
            nele = 0.0
            rho_int = 0.0
            ref_rho_int = 0.0
            for iat in range(natoms[iconf]):
                spe = atomic_symbols[iconf][iat]
                if spe in species:
                    nele += inp.pseudocharge
                    for l in range(lmax[spe]+1):
                        for n in range(nmax[(spe,l)]):
                            if l==0:
                                rho_int += charge_integrals[(spe,l,n)] * pred_coefs[iaux]
                                ref_rho_int += charge_integrals[(spe,l,n)] * ref_coefs[iaux]
                            iaux += 2*l+1
  
        # compute charge and dipole
        iaux = 0
        charge = 0.0
        dipole = 0.0
        ref_dipole = 0.0
        ref_charge = 0.0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe in species:
                if inp.average:
                    dipole += inp.pseudocharge * coords[iat,2]
                    ref_dipole += inp.pseudocharge * coords[iat,2]
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        for im in range(2*l+1):
                            if l==0 and im==0:
                                # rescale spherical coefficients to conserve the electronic charge
                                if inp.average:
                                    pred_coefs[iaux] *= nele/rho_int
                                charge += pred_coefs[iaux] * charge_integrals[(spe,l,n)]
                                dipole -= pred_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                                ref_charge += ref_coefs[iaux] * charge_integrals[(spe,l,n)]
                                ref_dipole -= ref_coefs[iaux] * charge_integrals[(spe,l,n)] * coords[iat,2]
                            if l==1 and im==1:
                                dipole -= pred_coefs[iaux] * dipole_integrals[(spe,l,n)]
                                ref_dipole -= ref_coefs[iaux] * dipole_integrals[(spe,l,n)]
                            iaux += 1
        print(iconf+1,ref_dipole,dipole,file=dfile)
        if inp.average:
            print(iconf+1,ref_charge,rho_int,file=qfile)
        else:
            print(iconf+1,ref_charge,charge,file=qfile)

    # save predicted coefficients
    np.save(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/prediction_conf"+str(iconf)+".npy",pred_coefs)

    # save predicted coefficients
    np.savetxt(inp.saltedpath+"validations_"+inp.saltedname+"/M"+str(M)+"_eigcut"+str(int(np.log10(eigcut)))+"/N"+str(ntrain)+"_reg"+str(int(np.log10(reg)))+"/COEFFS-"+str(iconf+1)+".dat",pred_coefs)

    # compute predicted density projections <phi|rho>
    pred_projs = np.dot(overl,pred_coefs)

    # compute error
    error = np.dot(pred_coefs-ref_coefs,pred_projs-ref_projs)
    error_density += error
    if inp.average:
        ref_projs -= np.dot(overl,Av_coeffs)
        ref_coefs -= Av_coeffs
    var = np.dot(ref_coefs,ref_projs)
    variance += var
    print(iconf+1,np.sqrt(error/var)*100,file=efile)
    print(iconf+1, ":", np.sqrt(error/var)*100, "% RMSE", flush=True)
    #print(iconf+1, ":", "rho integral =", rho_int, "normalized rho integral =", charge, "ref_dipole =", ref_dipole, "dipole =",dipole, ", error =", np.sqrt(error/var)*100, "% RMSE", flush=True)

    # UNCOMMENT TO CHECK PREDICTIONS OF <phi|rho-rho_0>
    # ------------------------------------------------- 
#    pred_projs = np.dot(overl,pred_coefs-Av_coeffs)
#    av_projs = np.dot(overl,Av_coeffs)
#    iaux = 0
#    for iat in range(natoms[iconf]):
#        spe = atomic_symbols[iconf][iat]
#        for l in range(lmax[spe]+1):
#            for n in range(nmax[(spe,l)]):
#                for im in range(2*l+1):
#                    if l==4 and im==0:
#                        print(pred_projs[iaux],ref_projs[iaux])
#                    iaux += 1

dfile.close()
print("")
print("% RMSE =", 100*np.sqrt(error_density/variance))
