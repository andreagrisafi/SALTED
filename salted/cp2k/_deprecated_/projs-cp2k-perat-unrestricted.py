import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
import copy
import argparse
import ctypes
import time

import pathlib
SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../../"
sys.path.append(SALTEDPATHLIB)
from lib import ovlp3c
from lib import ovlp3cXYperiodic
from lib import ovlp3cnonperiodic

SALTEDPATHLIB = str(pathlib.Path(__file__).parent.resolve())+"/../"
sys.path.append(SALTEDPATHLIB)
import basis

def cartesian_to_spherical_transformation(l):
        """Compute Cartesian to spherical transformation matrices sorting the spherical components as {-l,...,l} 
        while sorting the Cartesian components as shown in the corresponding arrays."""

        cart_idx = np.zeros((int((l+2)*(l+1)/2),3),int)
        if l==0:
            # 1 Cartesian triplet
            cart_idx = [[0,0,0]]
        elif l==1:
            # 3 Cartesian triplets
            cart_idx = [[1,0,0],[0,1,0],[0,0,1]]
        elif l==2:
            # 6 Cartesian triplets
            cart_idx = [[2,0,0],[1,1,0],[1,0,1],[0,2,0],[0,1,1],[0,0,2]]
        elif l==3:
            # 10 Cartesian triplets
            cart_idx = [[3,0,0],[2,1,0],[2,0,1],[1,2,0],[1,1,1],[1,0,2],[0,3,0],[0,2,1],[0,1,2],[0,0,3]]
        elif l==4:
            # 15 Cartesian triplets
            cart_idx = [[4,0,0],[3,1,0],[3,0,1],[2,2,0],[2,1,1],[2,0,2],[1,3,0],[1,2,1],[1,1,2],[1,0,3],[0,4,0],[0,3,1],[0,2,2],[0,1,3],[0,0,4]]        
        elif l==5:
            # 21 Cartesian triplets
            cart_idx = [[0,0,5],[2,0,3],[0,2,3],[4,0,1],[0,4,1],[2,2,1],[1,0,4],[0,1,4],[3,0,2],[0,3,2],[1,2,2],[2,1,2],[5,0,0],[0,5,0],[1,4,0],[4,1,0],[3,2,0],[2,3,0],[1,1,3],[3,1,1],[1,3,1]]
        elif l==6:
            # 28 Cartesian triplets
            cart_idx = [[6,0,0],[0,6,0],[0,0,6],[5,0,1],[5,1,0],[0,5,1],[1,5,0],[0,1,5],[1,0,5],[4,0,2],[4,2,0],[0,4,2],[2,4,0],[0,2,4],[2,0,4],[4,1,1],[1,4,1],[1,1,4],[3,1,2],[1,3,2],[1,2,3],[3,2,1],[2,3,1],[2,1,3],[3,3,0],[3,0,3],[0,3,3],[2,2,2]]
        else:
            print("ERROR: Cartesian to spherical transformation not available for l=",l)

        mat = np.zeros((2*l+1,int((l+2)*(l+1)/2)),complex)
        # this implementation follows Eq.15 of SCHLEGEL and FRISH, Inter. J. Quant. Chem., Vol. 54, 83-87 (1995)
        for m in range(l+1):
            itriplet = 0
            for triplet in cart_idx:
                lx = triplet[0]
                ly = triplet[1]
                lz = triplet[2]
                sfact = np.sqrt(math.factorial(l)*math.factorial(2*lx)*math.factorial(2*ly)*math.factorial(2*lz)*math.factorial(l-m)/(math.factorial(lx)*math.factorial(ly)*math.factorial(lz)*math.factorial(2*l)*math.factorial(l+m))) / (math.factorial(l)*2**l)
                j = (lx+ly-m)/2
                if j.is_integer()==True:
                    j = int(j)
                    if j>=0:
                       for ii in range(math.floor((l-m)/2)+1):
                           if j<=ii:
                               afact = special.binom(l,ii)*special.binom(ii,j)*math.factorial(2*l-2*ii)/math.factorial(l-m-2*ii)*(-1.0)**ii
                               for k in range(j+1):
                                   kk = lx-2*k
                                   if m>=kk and kk>=0:
                                      jj = (m-kk)/2
                                      bfact = special.binom(j,k)*special.binom(m,kk)*(-1.0)**(jj)
                                      mat[l+m,itriplet] += afact*bfact
                mat[l+m,itriplet] *= sfact 
                mat[l-m,itriplet] = np.conj(mat[l+m,itriplet]) 
                if m%2!=0:
                    mat[l+m,itriplet] *= -1.0 # TODO convention to be understood
                itriplet += 1
 
        return[np.asarray(mat), cart_idx]

def complex_to_real_transformation(sizes):
    """Transformation matrix from complex to real spherical harmonics"""
    matrices = []
    for i in range(len(sizes)):
        lval = int((sizes[i]-1)/2)
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range(lval):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[lval][lval] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)
    return matrices

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=1, help="Structure index")
    parser.add_argument("-kat", "--katom",  type=int, default=1, help="Atomic index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    kat = args.katom
    return [iconf,kat] 

args = add_command_line_arguments("")
[iconf,kat] = set_variable_values(args)

print("conf", iconf)
iconf -= 1 # 0-based indexing 
kat -= 1 # 0-based indexing 

bohr2angs = 0.529177210670

sys.path.insert(0, './')
import inp

qmpath = inp.path2qm

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)
species = inp.species
[lmax,nmax] = basis.basiset(inp.dfbasis)

# init geometry
geom = xyzfile[iconf]
geom.wrap()
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
contra = {}
for spe in species:
    with open(qmpath+"conf_"+str(iconf+1)+"/BASIS_MOLOPT") as f:
         for line in f:
             if line.rstrip().split()[0] == spe and line.rstrip().split()[1] == inp.qmbasis:
                line = list(islice(f, 2))[1]
                laomax[spe] = int(line.split()[2])
                npgf[spe] = int(line.split()[3])
                for l in range(laomax[spe]+1):
                    naomax[(spe,l)] = int(line.split()[4+l])
                    contra[(spe,l)] = np.zeros((naomax[(spe,l)],npgf[spe]))
                lines = list(islice(f, npgf[spe]))
                aoalphas[spe] = np.zeros(npgf[spe])
                for ipgf in range(npgf[spe]):
                    line = lines[ipgf].split()
                    aoalphas[spe][ipgf] = float(line[0])
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
    avals = np.loadtxt(qmpath+"conf_"+str(iconf+1)+"/alphas-"+str(spe)+".txt")
    for l in range(lmax[spe]+1):
        for n in range(nmax[(spe,l)]):
            alphas[(spe,l,n)] = avals[n] 
            sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr
            rcuts[(spe,l,n)] = np.sqrt(float(2+l)/(2*alphas[(spe,l,n)]))*4
    #with open("BASIS_LRIGPW_AUXMOLOPT") as f:
    #     for line in f:
    #         if line.rstrip().split()[0] == spe and line.rstrip().split()[-1] == inp.dfbasis:
    #            nalphas = int(list(islice(f, 1))[0])
    #            lines = list(islice(f, 1+2*nalphas))
    #            nval = {}
    #            for l in range(lmax[spe]+1):
    #                nval[l] = 0
    #            for ialpha in range(nalphas):
    #                alpha = np.array(lines[1+2*ialpha].split())[0]
    #                lbools = np.array(lines[1+2*ialpha].split())[1:]
    #                l = 0
    #                for ibool in lbools:
    #                    alphas[(spe,l,nval[l])] = float(alpha)
    #                    sigmas[(spe,l,nval[l])] = np.sqrt(0.5/alphas[(spe,l,nval[l])]) # bohr
    #                    rcuts[(spe,l,nval[l])] = np.sqrt(float(2+l)/(2*alphas[(spe,l,nval[l])]))*4
    #                    nval[l]+=1
    #                    l += 1
    #            break


# compute total number of auxiliary functions 
ntot = 0 
spe = symbols[kat]
for l in range(lmax[spe]+1):
    for n in range(nmax[(spe,l)]):
        ntot += 2*l+1
print("number of auxiliary functions =", ntot)

print("Reading AOs density matrix for spin alpha...")
nblocks = math.ceil(float(naotot)/2)
blocks = {}
for iblock in range(nblocks):
    blocks[iblock] = []
iblock = 0
for iao in range(naotot):
    blocks[math.floor(iao/2)].append(iao+1)
dm_alpha = np.zeros((naotot,naotot))
with open(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.dmfilealpha) as f:
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
                                dm_alpha[iao,iblock*2:iblock*2+len(dm_values)] = dm_values
                                iao += 1
         icount += 1

print("Reading AOs density matrix for spin beta...")
nblocks = math.ceil(float(naotot)/2)
blocks = {}
for iblock in range(nblocks):
    blocks[iblock] = []
iblock = 0
for iao in range(naotot):
    blocks[math.floor(iao/2)].append(iao+1)
dm_beta = np.zeros((naotot,naotot))
with open(inp.path2qm+"conf_"+str(iconf+1)+"/"+inp.dmfilebeta) as f:
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
                                dm_beta[iao,iblock*2:iblock*2+len(dm_values)] = dm_values
                                iao += 1
         icount += 1

dm = dm_alpha + dm_beta

# compute total number of Cartesian AOs
nao_cart = 0
for iat in range(natoms):
    spe = symbols[iat]
    for l in range(laomax[spe]+1):
        for n in range(naomax[spe,l]):
            nao_cart += int((1+l)*(2+l)/2)
print("Number of Cartesian atomic orbitals =",nao_cart)

# compute AOs normalization factor
nao_norm = {}
for spe in species:
    for l in range(laomax[spe]+1):
        for n in range(naomax[spe,l]):
            # include the normalization of primitive GTOs into the contraction coefficients
            for ipgf in range(npgf[spe]):
                prefac = 2.0**l*(2.0/np.pi)**0.75
                expalpha = 0.25*float(2*l+3)
                contra[(spe,l)][n,ipgf] *= prefac*aoalphas[spe][ipgf]**expalpha
            # compute inner product of contracted and normalized primitive GTOs
            nfact = 0.0
            for ipgf1 in range(npgf[spe]):
                for ipgf2 in range(npgf[spe]):
                    nfact += contra[(spe,l)][n,ipgf1] * contra[(spe,l)][n,ipgf2] * 0.5 * special.gamma(l+1.5) / ( (aoalphas[spe][ipgf1] + aoalphas[spe][ipgf2])**(l+1.5) )
            nao_norm[(spe,l,n)] = 1.0/np.sqrt(nfact)

# compute radial cutoffs for AOs
aorcuts = {}
for spe in species:
    for l in range(laomax[spe]+1):
        for ipgf in range(npgf[spe]):
            aorcuts[(spe,l,ipgf)] = np.sqrt(float(2+l)/(2*aoalphas[spe][ipgf]))*2

# define arrays for fortran
spemax = max(laomax,key=laomax.get)
nspecies = len(species)
laomaxx = np.zeros(nspecies,int)
npgfx = np.zeros(nspecies,int)
llaomax = laomax[spemax]
nnaomax = naomax[(spemax,0)]
naomaxx = np.zeros((nspecies,llaomax+1),int)
spemax = max(npgf,key=npgf.get)
npgfmax = npgf[spemax]
aoalphasx = np.zeros((nspecies,npgfmax))
contrax = np.zeros((nspecies,llaomax+1,nnaomax,npgfmax))
aorcutsx = np.zeros((nspecies,llaomax+1,npgfmax))
for ispe in range(nspecies):
    spe = species[ispe]
    laomaxx[ispe] = laomax[spe]
    npgfx[ispe] = npgf[spe]
    for l in range(laomax[spe]+1):
        naomaxx[ispe,l] = naomax[(spe,l)]
        for n in range(naomax[(spe,l)]):
            for ipgf in range(npgf[spe]):
                contrax[ispe,l,n,ipgf] = contra[(spe,l)][n,ipgf]
        for ipgf in range(npgf[spe]):
            aorcutsx[ispe,l,ipgf] = aorcuts[(spe,l,ipgf)]
    for ipgf in range(npgf[spe]):
        aoalphasx[ispe,ipgf] = aoalphas[spe][ipgf]
contrax = np.transpose(contrax,(3,2,1,0))
aorcutsx = np.transpose(aorcutsx,(2,1,0))

renorm = {}
for l in range(llaomax+1):
    triplets = cartesian_to_spherical_transformation(l)[1]
    renorm[l] = np.zeros((2*l+1,int((l+1)*(l+2)/2)))
    itriplet = 0
    for triplet in triplets:
        lx = triplet[0]
        ly = triplet[1]
        lz = triplet[2]
        renormfact = math.factorial(2*l+2) * math.factorial(lx) * math.factorial(ly) * math.factorial(lz)
        renormfact /= 8*np.pi * math.factorial(l+1) * math.factorial(2*lx) * math.factorial(2*ly) * math.factorial(2*lz)
        renorm[l][:,itriplet] = np.sqrt(renormfact)
        itriplet += 1

# get cartesian density matrix
dm_cart = np.zeros((nao_cart,nao_cart))
iao1 = 0
iao_cart1 = 0
for iat1 in range(natoms):
    spe1 = symbols[iat1]
    for l1 in range(laomax[spe1]+1):
        ncart1 = int((1+l1)*(2+l1)/2)
        c2r = complex_to_real_transformation([2*l1+1])[0]
        c2s_complex = cartesian_to_spherical_transformation(l1)[0]
        c2s_real1 = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[l1])
        for n1 in range(naomax[(spe1,l1)]):
            iao2 = 0
            iao_cart2 = 0
            for iat2 in range(natoms):
                spe2 = symbols[iat2]
                for l2 in range(laomax[spe2]+1):
                    ncart2 = int((1+l2)*(2+l2)/2)
                    c2r = complex_to_real_transformation([2*l2+1])[0]
                    c2s_complex = cartesian_to_spherical_transformation(l2)[0]
                    c2s_real2 = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[l2])
                    for n2 in range(naomax[(spe2,l2)]):
                        # convert density matrix in cartesian form
                        dm_cart[iao_cart1:iao_cart1+ncart1,iao_cart2:iao_cart2+ncart2] = np.dot(c2s_real1.T,np.dot(dm[iao1:iao1+2*l1+1,iao2:iao2+2*l2+1],c2s_real2))
                        # incorporate normalization of spherical atomic orbitals 
                        dm_cart[iao_cart1:iao_cart1+ncart1,iao_cart2:iao_cart2+ncart2] *= nao_norm[(spe1,l1,n1)] * nao_norm[(spe2,l2,n2)] 
                        iao2 += 2*l2+1
                        iao_cart2 += ncart2
            iao1 += 2*l1+1
            iao_cart1 += ncart1
#flatten down
dm_cart = dm_cart.reshape(nao_cart*nao_cart)

speidx = np.zeros(natoms,int) 
for iat in range(natoms):
    spe = symbols[iat]
    ispe = 1
    for spec in species:
        if spe==spec:
            speidx[iat] = ispe
        ispe += 1

llist = []
for spe in species:
    llist.append(lmax[spe])
llmax = max(llist)

cartidx = np.zeros((llmax+1,int((llmax+1)*(llmax+2)/2),3),int)
for l in range(llmax+1):
    triplets = cartesian_to_spherical_transformation(l)[1]
    cartidx[l][:int((l+1)*(l+2)/2)] = triplets
cartidx = np.transpose(cartidx,(2,1,0))

renorm = {}
for l in range(llmax+1):
    triplets = cartesian_to_spherical_transformation(l)[1]
    renorm[l] = np.zeros((2*l+1,int((l+1)*(l+2)/2)))
    itriplet = 0
    for triplet in triplets:
        lx = triplet[0] 
        ly = triplet[1] 
        lz = triplet[2]
        renormfact = math.factorial(2*l+2) * math.factorial(lx) * math.factorial(ly) * math.factorial(lz) 
        renormfact /= 8*np.pi * math.factorial(l+1) * math.factorial(2*lx) * math.factorial(2*ly) * math.factorial(2*lz)
        renorm[l][:,itriplet] = np.sqrt(renormfact)
        itriplet += 1

print("computing auxiliary density projections for atom", kat+1)
aux_projs = np.zeros(ntot) 
iaux = 0
start = time.time()
spe = symbols[kat]
print(spe)
print("-----------------")
coord = coords[kat].copy()
rr = np.dot(coord,coord)
for laux in range(lmax[spe]+1):
    print("angular momentum:",laux)
    ncart = int((2+laux)*(1+laux)/2)
    c2r = complex_to_real_transformation([2*laux+1])[0]
    c2s_complex = cartesian_to_spherical_transformation(laux)[0]
    c2s_real = np.multiply(np.real(np.dot(c2r,c2s_complex)),renorm[laux])
    for naux in range(nmax[(spe,laux)]):
        print("function:",naux+1,flush=True)
        # compute unnormalized 3-center overlap over cartesian functions (a|ij)
        if inp.periodic=="3D":
            ovlp_3c_cart = ovlp3c.ovlp3c(ncart,nao_cart,natoms,speidx,coords.T,cell,nspecies,llaomax,laomaxx,naomaxx,npgfx,rr,alphas[(spe,laux,naux)],npgfmax,aoalphasx,kat,laux,llmax,cartidx,nnaomax,contrax,rcuts[(spe,laux,naux)],aorcutsx) 
        elif inp.periodic=="2D":
            ovlp_3c_cart = ovlp3cXYperiodic.ovlp3c(ncart,nao_cart,natoms,speidx,coords.T,cell,nspecies,llaomax,laomaxx,naomaxx,npgfx,rr,alphas[(spe,laux,naux)],npgfmax,aoalphasx,kat,laux,llmax,cartidx,nnaomax,contrax,rcuts[(spe,laux,naux)],aorcutsx)
        elif inp.periodic=="0D":
            ovlp_3c_cart = ovlp3cnonperiodic.ovlp3c(ncart,nao_cart,natoms,speidx,coords.T,cell,nspecies,llaomax,laomaxx,naomaxx,npgfx,rr,alphas[(spe,laux,naux)],npgfmax,aoalphasx,kat,laux,llmax,cartidx,nnaomax,contrax,rcuts[(spe,laux,naux)],aorcutsx)
        else:
            print("ERROR: selected periodicity not implemented.")
            sys.exit(0) 
        ovlp_3c_cart = np.transpose(ovlp_3c_cart,(2,1,0))
        # flatten over AOs 
        ovlp_3c_cart = ovlp_3c_cart.reshape(ncart,nao_cart*nao_cart)
        # convert to spherical auxiliary functions 
        ovlp_3c = np.dot(c2s_real,ovlp_3c_cart) 
        # normalize auxiliary functions
        inner = 0.5*special.gamma(laux+1.5)*(sigmas[(spe,laux,naux)]**2)**(laux+1.5)
        ovlp_3c /= np.sqrt(inner)
        # compute density projections 
        aux_projs[iaux:iaux+2*laux+1] = np.dot(ovlp_3c,dm_cart)
        iaux += 2*laux+1
print("time =",(time.time()-start)/60.0,"minutes")

# save auxiliary projections
dirpath = os.path.join(qmpath, inp.projdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
np.save(qmpath+inp.projdir+"projections_conf"+str(iconf)+"_atom"+str(kat)+".npy",aux_projs)
