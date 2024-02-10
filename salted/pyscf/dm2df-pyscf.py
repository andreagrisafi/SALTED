import argparse
import os
import sys
import os.path as osp

import numpy as np
from pyscf import gto
from ase.io import read
from scipy import special

import basis  # WARNING: relative import

sys.path.insert(0, './')
import inp

def add_command_line_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-iconf", "--confidx",  type=int, default=-1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf 

args = add_command_line_arguments("")
iconf = set_variable_values(args)

xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

if iconf != -1:
    print("Calculating density matrix for configuration", iconf)
    iconf -= 1 # 0-based indexing
    conf_list = [iconf]
else:
    conf_list = range(ndata)

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)
    
dirpath = os.path.join(inp.path2qm, inp.coefdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, inp.projdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
dirpath = os.path.join(inp.path2qm, inp.ovlpdir)
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

print("PySCF orders all angular momentum components for L>1 as -L,...,0,...,+L,") 
print("and as +1,-1,0 for L=1, corresponding to X,Y,Z in Cartesian-SPH.")
print("Make sure to provide the density matrix following this convention!")
print("---------------------------------------------------------------------------------")
print("Reading geometry and basis sets...")

for iconf in conf_list:

    # Initialize geometry
    geom = xyzfile[iconf]
    symb = geom.get_chemical_symbols()
    coords = geom.get_positions()
    natoms = len(coords)
    atoms = []
    for i in range(natoms):
        coord = coords[i]
        atoms.append([symb[i],(coord[0],coord[1],coord[2])])
    
    # Get PySCF objects for wave-function and density-fitted basis
    mol = gto.M(atom=atoms,basis=inp.qmbasis)
    ribasis = inp.qmbasis+" jkfit"
    auxmol = gto.M(atom=atoms,basis=ribasis)
    pmol = mol + auxmol
    
    print("Computing overlap matrix...")
    
    # Get and save overlap matrix
    overlap = auxmol.intor('int1e_ovlp_sph')
    
    print("Computing density-fitted coefficients...")
    
    # Number of atomic orbitals
    nao = mol.nao_nr()
    # Number of auxiliary atomic orbitals
    naux = auxmol.nao_nr()
    # 2-centers 2-electrons integral
    eri2c = auxmol.intor('int2c2e_sph')
    # 3-centers 2-electrons integral
    eri3c = pmol.intor('int3c2e_sph', shls_slice=(0,mol.nbas,0,mol.nbas,mol.nbas,mol.nbas+auxmol.nbas))
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    # Load 1-electron reduced density-matrix
    dm=np.load(osp.join(
        inp.path2qm, "density_matrices", f"dm_conf{iconf+1}.npy"
    ))
    # Compute density fitted coefficients
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    
    print("Reordering...")
    
    # Reorder L=1 components following the -1,0,+1 convention
    Coef = np.zeros(len(rho),float)
    Over = np.zeros((len(rho),len(rho)),float)
    i1 = 0
    for iat in range(natoms):
        spe1 = symb[iat]
        for l1 in range(lmax[spe1]+1):
            for n1 in range(nmax[(spe1,l1)]):
                for im1 in range(2*l1+1):
                    if l1==1 and im1!=2:
                        Coef[i1] = rho[i1+1]
                    elif l1==1 and im1==2:
                        Coef[i1] = rho[i1-2]
                    else:
                        Coef[i1] = rho[i1]
                    i2 = 0
                    for jat in range(natoms):
                        spe2 = symb[jat]
                        for l2 in range(lmax[spe2]+1):
                            for n2 in range(nmax[(spe2,l2)]):
                                for im2 in range(2*l2+1):
                                    if l1==1 and im1!=2 and l2!=1:
                                        Over[i1,i2] = overlap[i1+1,i2]
                                    elif l1==1 and im1==2 and l2!=1:
                                        Over[i1,i2] = overlap[i1-2,i2]
                                    elif l2==1 and im2!=2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2+1]
                                    elif l2==1 and im2==2 and l1!=1:
                                        Over[i1,i2] = overlap[i1,i2-2]
                                    elif l1==1 and im1!=2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1+1,i2+1]
                                    elif l1==1 and im1!=2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1+1,i2-2]
                                    elif l1==1 and im1==2 and l2==1 and im2!=2:
                                        Over[i1,i2] = overlap[i1-2,i2+1]
                                    elif l1==1 and im1==2 and l2==1 and im2==2:
                                        Over[i1,i2] = overlap[i1-2,i2-2]
                                    else:
                                        Over[i1,i2] = overlap[i1,i2]
                                    i2 += 1
                    i1 += 1
    
    # Compute density projections on auxiliary functions
    Proj = np.dot(Over,Coef)
    
    # Save projections and overlaps
    np.save(f"inp.path2qm", "inp.coefdir", "coefficients_conf{iconf}.npy", Coef)
    np.save(f"inp.path2qm", "inp.projdir", "projections_conf{iconf}.npy", Proj)
    np.save(f"inp.path2qm", "inp.ovlpdir", "overlap_conf{iconf}.npy", Over)
    
    # --------------------------------------------------
    
    #print "Computing ab-initio energies.."
    #
    ## Hartree energy
    #J = np.einsum('Q,mnQ->mn', rho, eri3c)
    #e_h = np.einsum('ij,ji', J, dm) * 0.5
    #f = open("hartree_energy.dat", 'a') 
    #print >> f, e_h
    #f.close()
    #
    ## Nuclear-electron energy
    #h = mol.intor_symmetric('int1e_nuc')
    #e_Ne = np.einsum('ij,ji', h, dm) 
    #f = open("external_energy.dat", 'a') 
    #print >> f, e_Ne
    #f.close()
