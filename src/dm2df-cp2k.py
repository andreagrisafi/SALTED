import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
from scipy.interpolate import interp1d
import argparse
import ctypes
import time

import basis

from lib import gausslegendre
LIB = ctypes.CDLL(os.path.dirname(__file__) + "/lib/lebedev.so")
LDNS = [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434,
        590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890,
        4334, 4802, 5294, 5810]

def ld(n):
    '''Returns (x, y, z) coordinates along with weight w. Choose among [6, 14, 26, 38, 50, 74, 86, 110, 146, 170, 194, 230, 266, 302, 350, 434, 590, 770, 974, 1202, 1454, 1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810]'''

    if n not in LDNS:
        raise ValueError("n = {} not supported".format(n))
    xyzw = np.zeros((4, n), dtype=np.float64)
    c_double_p = ctypes.POINTER(ctypes.c_double)
    n_out = ctypes.c_int(0)
    getattr(LIB, "ld{:04}_".format(n))(
        xyzw[0].ctypes.data_as(c_double_p),
        xyzw[1].ctypes.data_as(c_double_p),
        xyzw[2].ctypes.data_as(c_double_p),
        xyzw[3].ctypes.data_as(c_double_p),
        ctypes.byref(n_out))
    assert n == n_out.value
    return xyzw.T


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

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

# get basis set info from CP2K BASIS_MOLOPT 
print("Reading AOs info...")
laomax = {}
naomax = {}
npgf = {}
aoalphas = {}
aosigmas = {}
aorcuts = {}
contra = {}
for spe in species:
    with open("BASIS_MOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == "DZVP-MOLOPT-GTH":
                line = list(islice(f, 2))[1]
                laomax[spe] = int(line.split()[2])
                npgf[spe] = int(line.split()[3])
                for l in range(laomax[spe]+1):
                    naomax[(spe,l)] = int(line.split()[4+l])
                    contra[(spe,l)] = np.zeros((naomax[(spe,l)],npgf[spe]))
                lines = list(islice(f, npgf[spe]))
                aoalphas[spe] = np.zeros(npgf[spe])
                aosigmas[spe] = np.zeros(npgf[spe])
                aorcuts[spe] = np.zeros(npgf[spe])
                for ipgf in range(npgf[spe]):
                    line = lines[ipgf].split()
                    aoalphas[spe][ipgf] = float(line[0])
                    aosigmas[spe][ipgf] = np.sqrt(0.5/aoalphas[spe][ipgf]) # bohr
                    aorcuts[spe][ipgf] = aosigmas[spe][ipgf]*100.0 # bohr
                    icount = 0
                    for l in range(laomax[spe]+1):
                        for n in range(naomax[(spe,l)]):
                            contra[(spe,l)][n,ipgf] = line[1+icount]
                            icount += 1  
                break
# compute total number of contracted atomic orbitals 
naotot = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(laomax[spe]+1):
        for n in range(naomax[(spe,l)]):
            naotot += 2*l+1
print("number of atomic orbitals =", naotot)

# get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
print("Reading auxiliary basis info...")
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
                        sigmas[(spe,l,nval[l])] = np.sqrt(0.5/alphas[(spe,l,nval[l])]) # bohr
                        rcuts[spe] = sigmas[(spe,l,nval[l])]*10.0 # bohr
                        nval[l]+=1
                        l += 1
                break
# compute total number of auxiliary functions 
ntot = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            ntot += 2*l+1
print("number of auxiliary functions =", ntot)

print("Reading AOs density matrix...")
nblocks = math.ceil(float(naotot)/4)
blocks = {}
for iblock in range(nblocks):
    blocks[iblock] = []
iblock = 0
for iao in range(naotot):
    blocks[math.floor(iao/4)].append(iao+1)

dm = np.zeros((naotot,naotot))
with open(inp.path2qm+"H2O-DM-1_0.Log") as f:
#with open(inp.path2qm+"runs/conf_"+str(iconf+1)+"/water-DM-1_0_1.Log") as f:
     icount = 1
     for line in f:
         if icount > 3:
            for iblock in range(nblocks):
                if len(line.rstrip().split())>0:
                    if int(line.rstrip().split()[0])==blocks[iblock][0] and int(line.rstrip().split()[-1])==blocks[iblock][-1]:
                        lines = list(islice(f, naotot+(natoms-1)))
                        iao = 0
                        for l in lines:
                            if len(l.split())>0:
                                dm_values = np.array(l.split()[4:]).astype(np.float)
                                dm[iao,iblock*4:iblock*4+len(dm_values)] = dm_values
                                iao += 1
         icount += 1

# compute interpolation function for contracted GTOs on a 1D radial mesh 
print("Precomputing radial part of AOs on a 1D mesh...")
ngrid = 10000
interp_radial_aos = {}
for spe in species:
    for l in range(laomax[spe]+1):
        for n in range(naomax[(spe,l)]):
            # include the normalization of primitive GTOs into the contraction coefficients
            for ipgf in range(npgf[spe]):
                prefac = 2.0**l*(2.0/np.pi)**0.75
                expalpha = 0.25*float(2*l + 3)
                contra[(spe,l)][n,ipgf] *= prefac*aoalphas[spe][ipgf]**expalpha
            # compute inner product of contracted and normalized primitive GTOs
            nfact = 0.0
            for ipgf1 in range(npgf[spe]):
                for ipgf2 in range(npgf[spe]):
                    nfact += contra[(spe,l)][n,ipgf1] * contra[(spe,l)][n,ipgf2] * 0.5 * special.gamma(l+1.5) / ( (aoalphas[spe][ipgf1] + aoalphas[spe][ipgf2])**(l+1.5) ) 
            # compute contracted radial functions
            rvec = np.zeros(ngrid)
            radial = np.zeros(ngrid)
            dxx = aorcuts[spe][-1]/float(ngrid-1)
            for ir in range(ngrid):
                r = ir*dxx
                rvec[ir] = r
                for ipgf in range(npgf[spe]):
                    radial[ir] += contra[(spe,l)][n,ipgf] * r**l * np.exp(-aoalphas[spe][ipgf]*r**2) 
            # normalize contracted radial functions 
            radial /= np.sqrt(nfact)
            # return interpolation function on 1D mesh 
            interp_radial_aos[(spe,l,n)] = interp1d(rvec,radial)

print("Constructing atom-centered grids and precomputing auxiliary functions...")
atomic_size = {}
atomic_grid = {}
atomic_weights = {}
atomic_functions = {}
for spe in species:
    # lebedev grid and integration weights
    lebsize = LDNS[lmax[spe]+4] 
    lebedev_grid = ld(lebsize)
    # init atomic grids
    atomic_size[spe] = lebsize*inp.radsize
    atomic_grid[spe] = np.zeros((atomic_size[spe],3),float)
    atomic_weights[spe] = np.zeros(atomic_size[spe])
    # Gauss-Legendre grid and integration weights
    gauss_points,gauss_weights = gausslegendre.gauss_legendre.gaulegf(x1=0.0,x2=rcuts[spe],n=inp.radsize)
    print(spe,"angular size =", lebsize, "radial size =", inp.radsize)   
    # precompute auxiliary functions 
    leb_grid = np.zeros((lebsize,3))
    leb_weights = np.zeros(lebsize)
    for ileb in range(lebsize):
        leb_grid[ileb,:] = lebedev_grid[ileb][:3]
        leb_weights[ileb] = lebedev_grid[ileb][3]
    lth = np.arccos(leb_grid[:,2])
    lph = np.arctan2(leb_grid[:,1],leb_grid[:,0])
    ylm_real = {}
    radial_aux = {}
    for l in range(lmax[spe]+1):
        # compute spherical harmonics on Lebedev
        ylm_real[l] = np.zeros((2*l+1,lebsize))
        lm = 0
        for m in range(-l,1):
            ylm = special.sph_harm(m,l,lph,lth)
            if m==0:
                ylm_real[l][lm,:] = np.real(ylm)/np.sqrt(2.0)
                lm += l+1
            else:
                ylm_real[l][lm,:] = -np.imag(ylm)
                ylm_real[l][lm-2*m,:] = np.real(ylm)
                lm += 1
        ylm_real[l] *= np.sqrt(2.0)
        for n in range(nmax[(spe,l)]):
            # init atomic functions
            atomic_functions[(spe,l,n)] = np.zeros((2*l+1,atomic_size[spe]))
            # compute radial functions on Gauss-Legendre
            radial_aux[(l,n)] = np.zeros(inp.radsize)
            inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5) 
            for irad in range(inp.radsize):
                r = gauss_points[irad]
                radial_aux[(l,n)][irad] = r**l * np.exp(-alphas[(spe,l,n)]*r**2) 
            radial_aux[(l,n)] /= np.sqrt(inner)
    # Construct atom centered grid
    igrid = 0
    for ileb in range(lebsize):
        for irad in range(inp.radsize):
            r = gauss_points[irad]
            atomic_grid[spe][igrid] = r * leb_grid[ileb] 
            atomic_weights[spe][igrid] = 4.0*np.pi * leb_weights[ileb] * r**2 * gauss_weights[irad]
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    atomic_functions[(spe,l,n)][:,igrid] = ylm_real[l][:,ileb] * radial_aux[(l,n)][irad]
            igrid += 1

print("Computing auxiliary density projections...")
iaux = 0
aux_projs = np.zeros(ntot)  
# for each atom used as center of the atomic grids
for icen in range(natoms):
    specen = symbols[icen]
    # move grid on the atomic center 
    agrid = atomic_grid[specen] + coords[icen]
    # initialize atomic orbitals on the atom-centered grid 
    aos = np.zeros((naotot,atomic_size[specen]))
    # loop over periodic images 
    for ix in [-1,0,1]:
        for iy in [-1,0,1]:
            for iz in[-1,0,1]:
                # collect contributions from each atom of the selected periodic image
                iao = 0
                for iat in range(natoms):
                    spe = symbols[iat] 
                    coord = coords[iat] 
                    #coord[0] -= cell[0,0]*round(coord[0]/cell[0,0])
                    #coord[1] -= cell[1,1]*round(coord[1]/cell[1,1])
                    #coord[2] -= cell[2,2]*round(coord[2]/cell[2,2])
                    coord[0] += ix*cell[0,0] 
                    coord[1] += iy*cell[1,1] 
                    coord[2] += iz*cell[2,2]
                    rr = agrid - coord
                    lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-10
                    lth = np.arccos(rr[:,2]/lr)
                    lph = np.arctan2(rr[:,1],rr[:,0]) 
                    for l in range(laomax[spe]+1):
                        # compute spherical harmonics on grid points
                        ylm_real = np.zeros((2*l+1,atomic_size[specen]))
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
                        for n in range(naomax[(spe,l)]):
                            #interpolate radial functions on grid points
                            radial_ao = interp_radial_aos[(spe,l,n)](lr)
                            #compute atomic orbitals
                            aos[iao:iao+2*l+1] += np.einsum("ab,b->ab",ylm_real,radial_ao)
                            iao += 2*l+1
    # compute electron density on the given atomic grid
    rho_r = np.zeros(atomic_size[specen])
    for iao1 in range(naotot):
        for iao2 in range(naotot):
            rho_r += np.multiply(aos[iao1],aos[iao2]) * dm[iao1,iao2]
    # multiply density by the integration weights
    rho_r *= atomic_weights[specen]
    # project electron density on the auxiliary functions
    for l in range(lmax[specen]+1):
        for n in range(nmax[(specen,l)]):
            # perform integration and store auxiliary projection
            aux_projs[iaux:iaux+2*l+1] = np.dot(atomic_functions[(specen,l,n)],rho_r) 
            iaux += 2*l+1

dirpath = os.path.join(inp.path2qm, "projections")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
#dirpath = os.path.join(inp.path2qm, "coefficients")
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)
#dirpath = os.path.join(inp.path2qm, "overlaps")
#if not os.path.exists(dirpath):
#    os.mkdir(dirpath)

# Save projections and overlaps
np.save(inp.path2qm+"projections/projections_conf"+str(iconf)+".npy",aux_projs)
#np.save(inp.path2qm+"coefficients/coefficients_conf"+str(iconf)+".npy",Coef)
#np.save(inp.path2qm+"overlaps/overlap_conf"+str(iconf)+".npy",Over)
