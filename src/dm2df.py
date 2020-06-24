import sys
import numpy as np
from pyscf import gto
from ase.io import read
from scipy import special
import argparse

def add_command_line_arguments(parsetext):
    parser = argparse.ArgumentParser(description=parsetext,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--confidx",  type=int, default=1, help="Structure index")
    args = parser.parse_args()
    return args

def set_variable_values(args):
    iconf = args.confidx
    return iconf 

args = add_command_line_arguments("")
iconf = set_variable_values(args)

# 0-based indexing
iconf -= 1 

import basis

sys.path.insert(0, './')
import inp

# read basis
[lmax,nmax] = basis.basiset(inp.basis)

print "PySCF orders all angular momentum components for L>1 as -L,...,0,...,+L," 
print "and as +1,-1,0 for L=1, corresponding to X,Y,Z in Cartesian-SPH."
print "Make sure to provide the density matrix following this convention!"
print "---------------------------------------------------------------------------------"
print "Reading geometry and basis sets..."


# Initialize geometry
geom = read("qm-runs/conf_"+str(iconf+1)+"/coords.xyz")
symb = geom.get_chemical_symbols()
coords = geom.get_positions()
natoms = len(coords)
atoms = []
for i in range(natoms):
    coord = coords[i]
    atoms.append([symb[i],(coord[0],coord[1],coord[2])])

# Get PySCF objects for wave-function and density-fitted basis
mol = gto.M(atom=atoms,basis="cc-pvqz")
auxmol = gto.M(atom=atoms,basis="cc-pvqz jkfit")
pmol = mol + auxmol

print "Computing overlap matrix..."

# Get and save overlap matrix
overlap = auxmol.intor('int1e_ovlp_sph')

print "Computing density-fitted coefficients..."

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
dm=np.load("qm-runs/conf_"+str(iconf+1)+"/density_matrix.npy")
# Compute density fitted coefficients
rho = np.einsum('ijp,ij->p', eri3c, dm)
rho = np.linalg.solve(eri2c, rho)

print "Reordering..."

# Reorder L=1 components following the -1,0,+1 convention
Coef = np.zeros(len(rho),float)
Over = np.zeros((len(rho),len(rho)),float)
i1 = 0
for iat in xrange(natoms):
    spe1 = symb[iat]
    for l1 in xrange(lmax[spe1]+1):
        for n1 in xrange(nmax[(spe1,l1)]):
            for im1 in xrange(2*l1+1):
                if l1==1 and im1!=2:
                    Coef[i1] = rho[i1+1]
                elif l1==1 and im1==2:
                    Coef[i1] = rho[i1-2]
                else:
                    Coef[i1] = rho[i1]
                i2 = 0
                for jat in xrange(natoms):
                    spe2 = symb[jat]
                    for l2 in xrange(lmax[spe2]+1):
                        for n2 in xrange(nmax[(spe2,l2)]):
                            for im2 in xrange(2*l2+1):
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
np.save("projections/projections_conf"+str(iconf)+".npy",Proj)
np.save("overlaps/overlap_conf"+str(iconf)+".npy",Over)

# --------------------------------------------------

print "Computing ab-initio electrostatic energy..."

# Hartree energy
J = np.einsum('Q,mnQ->mn', rho, eri3c)
e_h = np.einsum('ij,ji', J, dm) * 0.5

# Nuclear energy
E_NN=0
for I in range(auxmol.natm):   
    for J in range(I):
        if I != J:
            DIJ=np.linalg.norm(np.array(auxmol._atom[I][1])-np.array(auxmol._atom[J][1]))
            E_NN+=auxmol._atm[I][0]*auxmol._atm[J][0]/DIJ

# Nuclear-electron energy
h = mol.intor_symmetric('int1e_nuc')
e_Ne = np.einsum('ij,ji', h, dm) 

with open("qm-runs/conf_"+str(iconf+1)+"/QM-electro.dat", 'w') as file:
    file.write("E_NN: "+str(E_NN)+"\n")
    file.write("E_Ne: "+str(e_Ne)+"\n")
    file.write("E_ee: "+str(e_h)+"\n")
    file.write("Electrostatic energy:"+str(E_NN+e_Ne+e_h)+"\n")

# Density-fitted electrostatic energy
print "Computing density-fitted electrostatic energy..."

matrix_of_coefficients=np.outer(rho,rho)
e_h1=np.einsum('ij,ji',eri2c,matrix_of_coefficients)*0.5

def int_D0(symb,l,i):
    alpha=auxmol._basis[symb][i][1][0]
    sigma = np.sqrt(1.0/(2*alpha))
    inner = 0.5*special.gamma(l+1.5)*(sigma**2)**(l+1.5)
    return (0.5*(alpha**(-1-l/2))*special.gamma(1+l/2))/ np.sqrt(inner)

def int_D1(r,symb,l,i):
    alpha=auxmol._basis[symb][i][1][0]
    sigma = np.sqrt(1.0/(2*alpha))
    inner = 0.5*special.gamma(l+1.5)*(sigma**2)**(l+1.5)
    return ((r**(3 + 2*l)*(alpha*r**2)**(-1.5 - l)*(special.gamma(1.5 + l) - special.gamma(1.5 + l)*special.gammaincc(1.5 + l,alpha*r**2)))/2.)/np.sqrt(inner)

def int_D2(r,symb,l,i):
    alpha=auxmol._basis[symb][i][1][0]
    sigma = np.sqrt(1.0/(2*alpha))
    inner = 0.5*special.gamma(l+1.5)*(sigma**2)**(l+1.5)
    return (np.exp(-alpha*r**2)/(2*alpha))/np.sqrt(inner)

def complex_to_real_transformation(sizes):

    matrices = []
    for i in range(len(sizes)):
        lval = (sizes[i]-1)/2
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range( int((sizes[i]-1)/2) ):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[int((sizes[i]-1)/2)][int((sizes[i]-1)/2)] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)

    return matrices

def real_wigner(dij,lam):
    theta = np.arccos(dij[2]/np.linalg.norm(dij))
    phi = np.arctan2(dij[1],dij[0])
    wigner = np.zeros(2*lam+1,complex)
    for mu in range(2*lam+1):
        mu1=mu-lam
        wigner[mu] = np.sqrt(4*np.pi/(2*lam+1))*np.conj(special.sph_harm(mu1,lam,phi,theta))
    CC = np.conj(complex_to_real_transformation([2*lam+1])[0])
    return np.dot(CC,wigner)


nat=len(mol._atm)
i=0
j=0
coeffs=np.zeros(len(rho),dtype=float)
for iat in range(nat):
    atom_symb=auxmol._atom[iat][0]
    max_l=auxmol._basis[atom_symb][-1][0]
    numbs = [x[0] for x in auxmol._basis[auxmol._atom[iat][0]]]
    for l in range(max_l+1):
        n_l = numbs.count(l)
        for n in range(n_l):
            for m in range(2*l+1):
                if l ==1 :
                    if m != 0:
                        coeffs[i-1]=rho[i]
                        i+=1
                    else:
                        coeffs[i+2]=rho[i]
                        i+=1
                else:
                    coeffs[i]=rho[i]
                    i+=1
                

e_el=0
jdx=0
for I in range(auxmol.natm):   
    kdx=0
    contr3=0
    for J in range(auxmol.natm):
        if I==J:
            atm_symbol=auxmol._atom[J][0]
            lmax_J=auxmol._basis[atm_symbol][-1][0]
            numbs = [x[0] for x in auxmol._basis[atm_symbol]]
            idx=0
            for l in range(lmax_J+1):
                n_l = numbs.count(l)
                for n in range(n_l):
                    if l==0:
                        integral=int_D0(atm_symbol,l,idx)
                        idx+=1
                    else:
                        integral=0
                        idx+=1
                    for m in range(2*l+1):
                        contr3 += coeffs[jdx]*integral
                        jdx+=1
                        kdx+=1
        else:
            d_vec_IJ=np.array(auxmol._atom[I][1])-np.array(auxmol._atom[J][1])
            DIJ=np.linalg.norm(np.array(auxmol._atom[I][1])-np.array(auxmol._atom[J][1]))
            atm_symbol=auxmol._atom[J][0]
            lmax_J=auxmol._basis[atm_symbol][-1][0]
            numbs = [x[0] for x in auxmol._basis[atm_symbol]]
            idx=0
            for l in range(lmax_J+1):
                n_l = numbs.count(l)
                contr2=0
                for n in range(n_l):
                    integral1=int_D1(DIJ,atm_symbol,l,idx)
                    integral2=int_D2(DIJ,atm_symbol,l,idx)
    
                    integral1=integral1*1/(DIJ**(l+1))
                    integral2=integral2*(DIJ**(l))
            
                    idx+=1
                    W=np.real(real_wigner(d_vec_IJ,l))
                    contr = 0.0
                    for m in range(2*l+1):
                        contr+= coeffs[kdx]*W[m]
                        kdx+=1
                    contr2 += contr*(integral1+integral2)
                contr3 += contr2 * np.sqrt(1/(2*l+1))    
    e_el += contr3* auxmol._atm[I][0]

e_el=-e_el*np.sqrt(4*np.pi)

with open("qm-runs/conf_"+str(iconf+1)+"/RI-electro.dat", 'w') as file:
    file.write("E_NN: "+str(E_NN)+"\n")
    file.write("E_Ne: "+str(e_el)+"\n")
    file.write("E_ee: "+str(e_h1)+"\n")
    file.write("Electrostatic energy:"+str(E_NN+e_el+e_h1)+"\n")
