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

sys.path.insert(0, './')
import inp

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species

# init geometry
geom = xyzfile[iconf]
symbols = geom.get_chemical_symbols()
valences = geom.get_atomic_numbers()
coords = geom.get_positions()/bohr2angs
cell = geom.get_cell()/bohr2angs
natoms = len(coords)

# get basis set info from CP2K BASIS_MOLOPT 
laomax = {}
naomax = {}
nalphas = {}
alphas = {}
sigmas = {}
rcuts = {}
contra = {}
for spe in species:
    with open("BASIS_MOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == "DZVP-MOLOPT-GTH":
             #if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == "SZV-MOLOPT-GTH":
             #if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == "MINIMAL":
                line = list(islice(f, 2))[1]
                laomax[spe] = int(line.split()[2])
                nalphas[spe] = int(line.split()[3])
                for l in range(laomax[spe]+1):
                    naomax[(spe,l)] = int(line.split()[4+l])
                    contra[(spe,l)] = np.zeros((naomax[(spe,l)],nalphas[spe]))
                lines = list(islice(f, nalphas[spe]))
                alphas[spe] = np.zeros(nalphas[spe])
                sigmas[spe] = np.zeros(nalphas[spe])
                rcuts[spe] = np.zeros(nalphas[spe])
                for ipgf in range(nalphas[spe]):
                    line = lines[ipgf].split()
                    alphas[spe][ipgf] = float(line[0])
                    sigmas[spe][ipgf] = np.sqrt(0.5/alphas[spe][ipgf]) # bohr
                    rcuts[spe][ipgf] = sigmas[spe][ipgf]*10.0 # bohr
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

print("Reading density matrix...")
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
ngrid = 50000
interp_radial = {}
for spe in species:
    for l in range(laomax[spe]+1):
        for n in range(naomax[(spe,l)]):
            # include the normalization of primitive GTOs into the contraction coefficients
            for ipgf in range(nalphas[spe]):
                prefac = 2.0**l*(2.0/np.pi)**0.75
                expalpha = 0.25*float(2*l + 3)
                contra[(spe,l)][n,ipgf] *= prefac*alphas[spe][ipgf]**expalpha
            # compute inner product of contracted and normalized primitive GTOs
            nfact = 0.0
            for ipgf1 in range(nalphas[spe]):
                for ipgf2 in range(nalphas[spe]):
                    nfact += contra[(spe,l)][n,ipgf1] * contra[(spe,l)][n,ipgf2] * 0.5 * special.gamma(l+1.5) / ( (alphas[spe][ipgf1] + alphas[spe][ipgf2])**(l+1.5) ) 
            # compute contracted radial functions
            rvec = np.zeros(ngrid)
            radial = np.zeros(ngrid)
            dxx = rcuts[spe][-1]/float(ngrid-1)
            for ir in range(ngrid):
                r = ir*dxx
                rvec[ir] = r
                for ipgf in range(nalphas[spe]):
                    radial[ir] += contra[(spe,l)][n,ipgf] * r**l * np.exp(-alphas[spe][ipgf]*r**2) 
            # normalize contracted radial functions 
            radial /= np.sqrt(nfact)
            # return interpolation function on 1D mesh 
            interp_radial[(spe,l,n)] = interp1d(rvec,radial)

# define 3D grid
nside = {}
nside[0] = 45
nside[1] = 45
nside[2] = 45
npoints = 1
for i in range(3):
    npoints *= nside[i]
dx = cell[0,0] / nside[0]  # bohr 
dy = cell[1,1] / nside[1]  # bohr 
dz = cell[2,2] / nside[2]  # bohr 
origin = np.zeros(3)
grid_regular=np.transpose(np.asarray(np.meshgrid(dx*np.asarray(range(nside[0])),
                                                 dy*np.asarray(range(nside[1])),
                                                 dz*np.asarray(range(nside[2])) ) ),(2,1,3,0))
grid_regular=grid_regular.reshape((npoints,3))

# precompute contracted atomic orbitals on grid 
gtos = np.zeros((naotot,npoints))
#for ix in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
#    for iy in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
#        for iz in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
#for ix in [-4,-3,-2,-1,0,1,2,3,4]:
#    for iy in [-4,-3,-2,-1,0,1,2,3,4]:
#        for iz in [-4,-3,-2,-1,0,1,2,3,4]:
#for ix in [-3,-2,-1,0,1,2,3]:
#    for iy in [-3,-2,-1,0,1,2,3]:
#        for iz in[-3,-2,-1,0,1,2,3]:
for ix in [-1,0,1]:
    for iy in [-1,0,1]:
        for iz in[-1,0,1]:
#for ix in [-1,0,1]:
#    for iy in [-1,0,1]:
#        for iz in [-1,0,1]:
            iao = 0
            for iat in range(natoms):
                spe = symbols[iat] 
                coord = coords[iat] - origin
                #coord[0] -= cell[0,0]*round(coord[0]/cell[0,0])
                #coord[1] -= cell[1,1]*round(coord[1]/cell[1,1])
                #coord[2] -= cell[2,2]*round(coord[2]/cell[2,2])
                coord[0] += ix*cell[0,0] 
                coord[1] += iy*cell[1,1] 
                coord[2] += iz*cell[2,2]
                rr = grid_regular - coord
                lr = np.sqrt(np.sum(rr**2,axis=1)) + 1e-10 
                lth = np.arccos(rr[:,2]/lr)
                lph = np.arctan2(rr[:,1],rr[:,0])
                for l in range(laomax[spe]+1):
                    # compute spherical harmonics on grid points
                    ylm_real = np.zeros((2*l+1,npoints))
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
                        radial_gto = interp_radial[(spe,l,n)](lr)
                        #compute atomic orbitals
                        gtos[iao:iao+2*l+1] += np.einsum("ab,b->ab",ylm_real,radial_gto)
                        iao += 2*l+1

# compute density on grid
rho_r = np.zeros(npoints)
for iao1 in range(naotot):
    for iao2 in range(naotot):
        rho_r += np.multiply(gtos[iao1],gtos[iao2]) * dm[iao1,iao2] 

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
