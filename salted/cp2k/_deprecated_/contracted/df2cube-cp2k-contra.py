import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
from scipy.interpolate import interp1d
import argparse
import time

from sympy.parsing import mathematica
from sympy import symbols
from sympy import lambdify

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf 

args = add_command_line_arguments("")
iconf = set_variable_values(args)

#print("conf", iconf)
iconf -= 1 # 0-based indexing 

bohr2angs = 0.529177210670

import basis

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species

cubefilename = inp.dfcubefile
totcubefile = inp.totdfcubefile
totcharge = inp.totcharge

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

# get maximum values for lmax and nmax
llist = []
nlist = []
for spe in species:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# init geometry
geom = xyzfile[iconf]
geom.wrap()
chemical_symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT 
alphas = {}
sigmas = {}
rcuts = {}
for spe in species:
    avals = np.loadtxt("alphas-"+str(spe)+".txt")
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            alphas[(spe,l,n)] = avals[n]
            sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
            rcuts[(spe,l,n)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,n)]))*4
            rcutmax = np.sqrt(float(2+l)/(2*alphas[(spe,l,n)]))*4

# If total charge density is asked, read in the GTH pseudo-charge and return a radial numpy function
if totcharge:
    pseudof = open(inp.pseudochargefile,"r")
    pseudocharge = mathematica.mathematica(pseudof.readlines()[0],{'Erf[x]':'erf(x)'})
    pseudof.close()
    rpseudo = symbols('r')
    pseudocharge = lambdify(rpseudo, pseudocharge, modules=['numpy'])
    pseudocharge = np.vectorize(pseudocharge)
    nn = 100000
    dr = 5.0/nn
    charge = 0.0
    for ir in range(1,nn):
        r = ir*dr
        charge += r**2*pseudocharge(r)
    print("Integrated pseudo-charge =", 4*np.pi*charge*dr)


naux = 0
for iat in range(natoms):
    spe = chemical_symbols[iat]
    for l in range(lmax[spe]+1):
         naux += nmax[(spe,l)]*(2*l+1)
print("number of primitive auxiliary functions =", naux)

# load RI-coefficients
coefs = np.load(inp.path2qm+inp.coefdir+"coefficients_conf"+str(iconf)+".npy")
print("number of contracted auxiliary functions =", len(coefs))

# compute GTOs on a 1D radial mesh 
ngrid = 20000
ncut = {}
interp_radial = {}
for spe in species:
    for l in range(lmax[spe]+1):
        projector = np.load("contractions/contra_spe"+str(spe)+"_l"+str(l)+".npy")
        ncut[(spe,l)] = projector.shape[-1]
        radial = np.zeros((nmax[(spe,l)],ngrid))
        for n in range(nmax[(spe,l)]):
            dxx = rcutmax/float(ngrid-1)
            rvec = np.zeros(ngrid)
            for irad in range(ngrid):
                r = irad*dxx
                rvec[irad] = r
                radial[n,irad] = r**l * np.exp(-alphas[(spe,l,n)]*r**2)
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            radial[n] /= np.sqrt(inner)
        contr_radial = np.dot(projector.T,radial)
#        np.savetxt("radials/radial_points.txt",rvec)
#        np.savetxt("radials/contr_radials_spe"+str(spe)+"_l"+str(l)+".txt",contr_radial)
        for n in range(ncut[(spe,l)]):
            interp_radial[(spe,l,n)] = interp1d(rvec,contr_radial[n],kind="quadratic")

cubefile = open(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.cubefile ,"r")
lines = cubefile.readlines()
nside = {}
nside[0] = int(lines[3].split()[0])
nside[1] = int(lines[4].split()[0])
nside[2] = int(lines[5].split()[0])
npoints = 1
for i in range(3):
    npoints *= nside[i]
dx = float(lines[3].split()[1])
dy = float(lines[4].split()[2])
dz = float(lines[5].split()[3])
origin = np.asarray(lines[2].split(),dtype=float)[1:4]
rho_qm = []
for line in lines[6+natoms:]:
    rhovals = np.asarray(line.split(),float)
    for rhoval in rhovals:
        rho_qm.append(rhoval)
cubefile.close()

nele = np.sum(rho_qm)*dx*dy*dz
print("reference number of electrons=",nele)

if npoints!=len(rho_qm):
   print("ERROR: inconsistent number of grid points!")
   sys.exit(0)

grid_regular=np.transpose(np.asarray(np.meshgrid(dx*np.asarray(range(nside[0])),
                                                 dy*np.asarray(range(nside[1])),
                                                 dz*np.asarray(range(nside[2])) ) ),(2,1,3,0))
grid_regular=grid_regular.reshape((npoints,3))

# compute 3D density
rho_r = np.zeros(npoints)
if totcharge:
    pseudo_charge = np.zeros(npoints)
naux_old = 0
for iat in range(natoms):
    print(iat+1,flush=True)
    spe = chemical_symbols[iat]
    repmax = np.zeros(3,int)
    for ix in range(3):
        nreps = math.ceil(rcutmax/cell[ix,ix])
        if nreps < 1:
            repmax[ix] = 1
        else:
            repmax[ix] = nreps
    naux = 0
    for l in range(lmax[spe]+1):
        for n in range(ncut[(spe,l)]):
            naux += 2*l+1
    coef = coefs[naux_old:naux_old+naux]
    naux_old += naux
    if inp.periodic=="3D":
        for ix in range(-repmax[0],repmax[0]+1):
            for iy in range(-repmax[1],repmax[1]+1):
                for iz in range(-repmax[2],repmax[2]+1):
                    coord = coords[iat] - origin
                    coord[0] += ix*cell[0,0]
                    coord[1] += iy*cell[1,1]
                    coord[2] += iz*cell[2,2]
                    dvec = grid_regular - coord
                    d2list = np.sum(dvec**2,axis=1)
                    indx = np.where(d2list<=rcutmax**2)[0]
                    nidx = len(indx)
                    rr = dvec[indx]
                    lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
                    if totcharge:
                        if nidx>0:
                            pseudo_charge[indx] += pseudocharge(lr)
                    lth = np.arccos(rr[:,2]/lr)
                    lph = np.arctan2(rr[:,1],rr[:,0])
                    iaux = 0
                    for l in range(lmax[spe]+1):
                        ylm_real = np.zeros((2*l+1,nidx))
                        lm = 0
                        for m in range(-l,1):
                            ylm = special.sph_harm(m,l,lph,lth)
                            if m==0:
                                ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                                lm += l+1
                            else:
                                ylm_real[lm,:] = -np.imag(ylm)
                                ylm_real[lm-2*m,:] = np.real(ylm)
                                lm += 1
                        ylm_real *= np.sqrt(2.0)
                        for n in range(ncut[(spe,l)]):
                            #interpolate radial functions
                            radial_gto = interp_radial[(spe,l,n)](lr)
                            gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                            rho_r[indx] += np.dot(coef[iaux:iaux+2*l+1],gto)
                            iaux += 2*l+1
    elif inp.periodic=="2D":
        for ix in range(-repmax[0],repmax[0]+1):
            for iy in range(-repmax[1],repmax[1]+1):
                coord = coords[iat] - origin
                coord[0] += ix*cell[0,0]
                coord[1] += iy*cell[1,1]
                dvec = grid_regular - coord
                d2list = np.sum(dvec**2,axis=1)
                indx = np.where(d2list<=rcutmax**2)[0]
                nidx = len(indx)
                rr = dvec[indx]
                lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
                if totcharge:
                    if nidx>0:
                        pseudo_charge[indx] += pseudocharge(lr)
                lth = np.arccos(rr[:,2]/lr)
                lph = np.arctan2(rr[:,1],rr[:,0])
                iaux = 0
                for l in range(lmax[spe]+1):
                    ylm_real = np.zeros((2*l+1,nidx))
                    lm = 0
                    for m in range(-l,1):
                        ylm = special.sph_harm(m,l,lph,lth)
                        if m==0:
                            ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                            lm += l+1
                        else:
                            ylm_real[lm,:] = -np.imag(ylm)
                            ylm_real[lm-2*m,:] = np.real(ylm)
                            lm += 1
                    ylm_real *= np.sqrt(2.0)
                    for n in range(ncut[(spe,l)]):
                        #interpolate radial functions
                        radial_gto = interp_radial[(spe,l,n)](lr)
                        gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                        rho_r[indx] += np.dot(coef[iaux:iaux+2*l+1],gto)
                        iaux += 2*l+1
    elif inp.periodic=="0D":
        coord = coords[iat] - origin
        dvec = grid_regular - coord
        d2list = np.sum(dvec**2,axis=1)
        indx = np.where(d2list<=rcutmax**2)[0]
        nidx = len(indx)
        rr = dvec[indx]
        lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-12
        if totcharge:
            if nidx>0:
                pseudo_charge[indx] += pseudocharge(lr)
        lth = np.arccos(rr[:,2]/lr)
        lph = np.arctan2(rr[:,1],rr[:,0])
        iaux = 0
        for l in range(lmax[spe]+1):
            ylm_real = np.zeros((2*l+1,nidx))
            lm = 0
            for m in range(-l,1):
                ylm = special.sph_harm(m,l,lph,lth)
                if m==0:
                    ylm_real[lm,:] = np.real(ylm)/np.sqrt(2.0)
                    lm += l+1
                else:
                    ylm_real[lm,:] = -np.imag(ylm)
                    ylm_real[lm-2*m,:] = np.real(ylm)
                    lm += 1
            ylm_real *= np.sqrt(2.0)
            for n in range(ncut[(spe,l)]):
                #interpolate radial functions
                radial_gto = interp_radial[(spe,l,n)](lr)
                gto = np.einsum("ab,b->ab",ylm_real,radial_gto)
                rho_r[indx] += np.dot(coef[iaux:iaux+2*l+1],gto)
                iaux += 2*l+1
    else:
        print("ERROR: selected periodicity not implemented.")
        sys.exit(0)

# compute integrated electronic charge
nele = np.sum(rho_r)*dx*dy*dz
print("number of electrons=",nele)

# compute error as a fraction of electronic charge
error = np.sum(abs(rho_r-rho_qm))*dx*dy*dz/nele
print("% error =", error*100)

# pring density on a cube file
cubef = open(inp.path2qm+"conf_"+str(iconf+1)+"/"+cubefilename,"w")
print("Reconstructed electron density",file=cubef)
print("CUBE FORMAT",file=cubef)
print(natoms, origin[0], origin[1], origin[2],file=cubef)
metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
for ix in range(3):
    print(nside[ix], metric[ix,0], metric[ix,1], metric[ix,2],file=cubef)
for iat in range(natoms):
    print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
for igrid in range(npoints):
    print(rho_r[igrid],file=cubef)
cubef.close()

if totcharge:
    rho_Q = pseudo_charge - rho_r
    print("total charge =",np.sum(rho_Q)*dx*dy*dz)
    # pring density on a cube file
    cubef = open(inp.path2qm+"conf_"+str(iconf+1)+"/"+totcubefile,"w")
    print("Reconstructed electron density",file=cubef)
    print("CUBE FORMAT",file=cubef)
    print(natoms, origin[0], origin[1], origin[2],file=cubef)
    metric = np.array([[dx,0.0,0.0],[0.0,dy,0.0],[0.0,0.0,dz]])
    for ix in range(3):
        print(nside[ix], metric[ix,0], metric[ix,1], metric[ix,2],file=cubef)
    for iat in range(natoms):
        print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2],file=cubef)
    for igrid in range(npoints):
        print(rho_Q[igrid],file=cubef)
    cubef.close()

