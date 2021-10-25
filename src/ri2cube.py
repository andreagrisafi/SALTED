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

print("conf", iconf)
iconf -= 1 # 0-based indexing 

bohr2angs = 0.529177249

import basis

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species

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
symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

# load RI-coefficients
coefs = np.load(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy")

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT 
alphas = {}
sigmas = {}
rcuts = {}
for spe in species:
    with open("BASIS_LRIGPW_AUXMOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[-1] == "LRI-DZVP-MOLOPT-GTH-MEDIUM":
                nalphas = int(list(islice(f, 1))[0])
                lines = list(islice(f, 1+2*nalphas))
                nval = {}
                for l in range(lmax[spe]+1):
                    nval[l] = 0
                for ialpha in range(nalphas):
                    alpha = np.array(lines[1+2*ialpha].split())[0]
                    lbools = np.array(lines[1+2*ialpha].split())[1:]
                    l = 0
                    for ibool in lbools:
                        alphas[(spe,l,nval[l])] = float(alpha)
                        sigmas[(spe,l,nval[l])] = np.sqrt(0.5/alphas[(spe,l,nval[l])]) # bohr assumed here! 
                        rcuts[(spe,l,nval[l])] = sigmas[(spe,l,nval[l])]*10.0 # bohr
                        nval[l]+=1
                        l += 1
                break

def radial_GTO(rval,alphaval,lval):
    """Evaluate radial part of Gaussian type orbitals"""
    f = (rval**lval)*np.exp(-alphaval*(rval)**2)
    return f

# compute GTOs on a 1D radial mesh #TODO this can be done once and for all
ngrid = 10000
rvec = {}
radial = {}
for spe in species:
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            rvec[(spe,l,n)] = np.zeros(ngrid)
            radial[(spe,l,n)] = np.zeros(ngrid)
            dxx = rcuts[(spe,l,n)]/float(ngrid-1)
            #inner = np.sqrt(np.pi)*math.factorial(2*l+2)/(2.0**(l+3)*math.factorial(l+1)*(2*alphas[(spe,l,n)])**(l+1.5))
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
            for ir in range(ngrid):
                r = ir*dxx
                rvec[(spe,l,n)][ir] = r
                radial[(spe,l,n)][ir] = radial_GTO(r,alphas[(spe,l,n)],l)
            radial[(spe,l,n)] /= np.sqrt(inner)

# define 3D grid
nside = {}
nside[0] = 50 
nside[1] = 50
nside[2] = 50
npoints = 1
for i in range(3):
    npoints *= nside[i]
dx = cell[0,0] / nside[0]  # bohr 
dy = cell[1,1] / nside[1]  # bohr 
dz = cell[2,2] / nside[2]  # bohr 
origin = np.array([0.0,0.0,0.0])#np.array([cell[0,0]/2.0,cell[1,1]/2.0,cell[2,2]/2.0]) # bohr
grid_regular=np.transpose(np.asarray(np.meshgrid(dx*np.asarray(range(nside[0])),
                                                 dy*np.asarray(range(nside[1])),
                                                 dz*np.asarray(range(nside[2])) ) ),(2,1,3,0))
grid_regular=grid_regular.reshape((npoints,3))

# compute 3D density
rho_r = np.zeros(npoints)
for ix in [-2,-1,0,1,2]:
    for iy in [-2,-1,0,1,2]:
        for iz in [-2,-1,0,1,2]:
            itot=0 
            for iat in range(natoms):
                coord = coords[iat] - origin
                coord[0] += ix*cell[0,0] 
                coord[1] += iy*cell[1,1] 
                coord[2] += iz*cell[2,2] 
                dvec = grid_regular - coord
                d2list = np.sum(dvec**2,axis=1)
                spe = symbols[iat] 
                for l in range(lmax[spe]+1):
                    for n in range(nmax[(spe,l)]):
                        #grid per channel
                        indx = np.where(d2list<=rcuts[(spe,l,n)]**2)[0]
                        nidx = len(indx)
                        rr = dvec[indx] 
                        #interpolate radial functions
                        lr = np.sqrt(np.sum(rr**2,axis=1))
                        interp_radial = interp1d(rvec[(spe,l,n)],radial[(spe,l,n)])
                        radial_gto = interp_radial(lr)            
                        #harmonics
                        lth = np.arccos(rr[:,2]/lr)
                        lph = np.arctan2(rr[:,1],rr[:,0])
                        ylm_real = np.zeros((nidx,2*l+1))
                        lm = 0
                        for m in range(-l,1):
                            ylm = special.sph_harm(m,l,lph,lth)
                            if m==0:
                                ylm_real[:,lm] = np.real(ylm)/np.sqrt(2.0)
                                lm += l+1
                            else:
                                ylm_real[:,lm] = -np.imag(ylm)
                                ylm_real[:,lm-2*m] = np.real(ylm)
                                lm += 1
                        ylm_real *= np.sqrt(2.0) 
                        gto = np.einsum("a,ab->ab",radial_gto,ylm_real)
                        rho_r[indx] += np.dot(gto,coefs[itot:itot+2*l+1])
                        itot += 2*l+1 

print("number of electrons=",np.sum(rho_r)*dx*dy*dz)

dirpath = os.path.join(inp.path2qm, "cubes")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# pring density on a cube file
filename = "cubes/rhor_conf-"+str(iconf+1)+".cube"
cubef = open(inp.path2qm+filename,"w")
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
