import argparse
import os
import sys
import os.path as osp
from typing import List, Tuple, Union, Dict
import time

import numpy as np
from pyscf import gto, df, lib
from pyscf.gto import basis
from ase.io import read
from scipy import special

from salted.basis_client import BasisClient, SpeciesBasisData
from salted.sys_utils import ParseConfig, parse_index_str, ARGHELP_INDEX_STR, Irreps


__doc__ = """
PySCF orders all angular momentum components for L>1 as -L,...,0,...,+L,
and as +1,-1,0 for L=1, corresponding to X,Y,Z in Cartesian-SPH.
For details please check https://pyscf.org/user/gto.html#ordering-of-basis-functions
Make sure to provide the density matrix following this convention!
"""

# pyscf_time = 0.0
# reorder_time = 0.0


def cal_df_coeffs_old(
    atoms: List,
    qmbasis: str,
    ribasis: str,
    dm: np.ndarray,
    lmax: Dict[str, int],
    nmax: Dict[Tuple[str, int], int],
):
    # global reorder_time, pyscf_time

    # start_time = time.time()
    mol = gto.M(atom=atoms, basis=qmbasis)
    auxmol = gto.M(atom=atoms, basis=ribasis)
    pmol = mol + auxmol
    assert dm.shape[0] == mol.nao_nr(), f"{dm.shape=}, {mol.nao_nr()=}"

    overlap = auxmol.intor("int1e_ovlp_sph")  # AO overlap matrix
    eri2c = auxmol.intor('int2c2e_sph')  # 2-centers 2-electrons integral
    eri3c = pmol.intor(  # 3-centers 2-electrons integral
        'int3c2e_sph',
        shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    )
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    # pyscf_time += time.time() - start_time

    # start_time = time.time()
    # Reorder L=1 components following the -1,0,+1 convention
    Coef = np.zeros(len(rho))
    Over = np.zeros((len(rho), len(rho)))

    i1 = 0
    for iat in range(len(atoms)):
        spe1 = atoms[iat][0]
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
                    for jat in range(len(atoms)):
                        spe2 = atoms[jat][0]
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
    # reorder_time += time.time() - start_time

    return Coef, Proj, Over


def cal_df_coeffs(
    atoms: List,
    qmbasis: str,
    ribasis: str,
    dm: np.ndarray,
    irreps: Irreps,
):
    # global reorder_time, pyscf_time

    # start_time = time.time()
    mol = gto.M(atom=atoms, basis=qmbasis)
    auxmol = gto.M(atom=atoms, basis=ribasis)
    pmol = mol + auxmol
    assert dm.shape[0] == mol.nao_nr(), f"{dm.shape=}, {mol.nao_nr()=}"

    overlap = auxmol.intor("int1e_ovlp_sph")  # AO overlap matrix
    eri2c = auxmol.intor('int2c2e_sph')  # 2-centers 2-electrons integral
    eri3c = pmol.intor(  # 3-centers 2-electrons integral
        'int3c2e_sph',
        shls_slice=(0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    )
    eri3c = eri3c.reshape(mol.nao_nr(), mol.nao_nr(), -1)
    rho = np.einsum('ijp,ij->p', eri3c, dm)
    rho = np.linalg.solve(eri2c, rho)
    # pyscf_time += time.time() - start_time

    """ Reorder L=1 components from +1,-1,0 to -1,0,+1 """
    # start_time = time.time()
    l1_slices = irreps.slices_l(1)
    l1_indexes = np.array([sl.start for sl in l1_slices]).reshape(-1, 1) + np.array([0, 1, 2]).reshape(1, -1)  # shape (n_l1, 3)
    l1_indexes_from = l1_indexes.reshape(-1)  # shape (3*n_l1,)
    l1_indexes_to = l1_indexes[:, [1, 2, 0]].reshape(-1)  # shape (3*n_l1,)

    coef_reordered = rho
    coef_reordered[l1_indexes_to] = rho[l1_indexes_from]
    overlap_reordered = overlap
    overlap_reordered[l1_indexes_to] = overlap[l1_indexes_from]
    overlap_reordered[:, l1_indexes_to] = overlap[:, l1_indexes_from]

    # Compute density projections on auxiliary functions
    proj_reordered = np.dot(overlap_reordered, coef_reordered)
    # reorder_time += time.time() - start_time

    return coef_reordered, proj_reordered, overlap_reordered

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

def main(geom_indexes: Union[List[int], None], num_threads: int = None):
    # global reorder_time, pyscf_time
    inp = ParseConfig().parse_input()
    geoms_all = read(inp.system.filename, ":")
    if geom_indexes is None:
        geom_indexes = list(range(len(geoms_all)))
    else:
        geom_indexes = [i for i in geom_indexes if i < len(geoms_all)]  # indexes start from 0
    print(f"Calculate density fitting coefficients for these structures: {geom_indexes}")
    geoms = [geoms_all[i] for i in geom_indexes]

    """ prepare directories to store data """
    for data_dname in ("coefficients", "projections", "overlaps"):
        if not osp.exists(dpath := osp.join(inp.qm.path2qm, data_dname)):
            os.mkdir(dpath)

    """ set pyscf.lib.num_threads """
    if num_threads is not None:
        lib.num_threads(num_threads)

    # lmax, nmax = BasisClient().read_as_old_format(inp.qm.dfbasis)
    ribasis = df.addons.DEFAULT_AUXBASIS[basis._format_basis_name(inp.qm.qmbasis)][0]  # RI basis name in pyscf
    basis_data = BasisClient().read(inp.qm.dfbasis)
    df_irreps_by_spe = {
        spe: Irreps(tuple((cnt, lam) for lam, cnt in enumerate(spe_basis_data["nmax"])))
        for spe, spe_basis_data in basis_data.items()
    }

    """ do density fitting """
    start_time = time.time()
    for cal_idx, (geom_idx, geom) in enumerate(zip(geom_indexes, geoms)):
        print(f"calculate {geom_idx=}, progress: {cal_idx}/{len(geom_indexes)}")
        symb = geom.get_chemical_symbols()
        coords = geom.get_positions()
        atoms = [(s, c) for s, c in zip(symb, coords)]
        irreps = sum([df_irreps_by_spe[spe] for spe in symb], Irreps([]))
        dm = np.load(osp.join(inp.qm.path2qm, "density_matrices", f"dm_conf{geom_idx+1}.npy"))
        # coef, proj, over = cal_df_coeffs_old(atoms, inp.qm.qmbasis, ribasis, dm, lmax, nmax)
        coef, proj, over = cal_df_coeffs(atoms, inp.qm.qmbasis, ribasis, dm, irreps)
        np.save(osp.join(inp.qm.path2qm, "coefficients", f"coefficients_conf{geom_idx}.npy"), coef)
        np.save(osp.join(inp.qm.path2qm, "projections", f"projections_conf{geom_idx}.npy"), proj)
        np.save(osp.join(inp.qm.path2qm, "overlaps", f"overlap_conf{geom_idx}.npy"), over)
    end_time = time.time()
    print(f"Calculation finished, time cost on density fitting: {end_time - start_time:.2f}s")
    # print(f"Time cost on PySCF: {pyscf_time:.2f}s, on reordering: {reorder_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # create a parser obj, which accepts the indexes to calculate, start from 0
    # formats: 1,2,3 or 1-3 or None (all structures)
    parser.add_argument(
        "-i", "--idx", type=str, default="all",
        help=ARGHELP_INDEX_STR,
    )
    parser.add_argument(
        "-c", "--cpu", type=int, default=None,
        help="Number of CPU cores to use. Default is None (for do nothing)."
    )
    args = parser.parse_args()

    """ pyscf.lib.num_threads is more important than numpy OMP"""
    # # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy
    # if args.cpu is not None:
    #     assert isinstance(args.cpu, int) and args.cpu > 0, f"{args.cpu=}"
    #     for env_var in (
    #         "OMP_NUM_THREADS",
    #         "OPENBLAS_NUM_THREADS",
    #         "MKL_NUM_THREADS",
    #         "VECLIB_MAXIMUM_THREADS",
    #         "NUMEXPR_NUM_THREADS",
    #     ):
    #         os.environ[env_var] = str(args.cpu)

    main(parse_index_str(args.idx), args.cpu)
