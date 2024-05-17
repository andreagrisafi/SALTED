import argparse
import os
import sys
import time
from typing import List, Tuple, Union

import numpy as np
from ase.io import read
from pyscf import dft, gto, lib

from salted.sys_utils import ARGHELP_INDEX_STR, ParseConfig, parse_index_str


def run_pyscf(
    atoms: List,
    basis: str,
    xc: str,
):
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.kernel()
    return mf.make_rdm1()


def main(geom_indexes: Union[List[int], None], num_threads: int = None):
    inp = ParseConfig().parse_input()
    geoms_all = read(inp.system.filename, ":")
    if geom_indexes is None:
        geom_indexes = list(range(len(geoms_all)))
    else:
        geom_indexes = [i for i in geom_indexes if i < len(geoms_all)]  # indexes start from 0
    print(f"Calculating density matrix for configurations: {geom_indexes}")
    geoms = [geoms_all[i] for i in geom_indexes]

    """ prepare the output directory """
    dirpath = os.path.join(inp.qm.path2qm, "density_matrices")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    """ set pyscf.lib.num_threads """
    if num_threads is not None:
        lib.num_threads(num_threads)

    """ do DFT calculation """
    start_time = time.time()
    for cal_idx, (geom_idx, geom) in enumerate(zip(geom_indexes, geoms)):
        print(f"calcualte {geom_idx=}, progress: {cal_idx}/{len(geom_indexes)}")
        symb = geom.get_chemical_symbols()
        coords = geom.get_positions()
        atoms = [(s, c) for s, c in zip(symb, coords)]

        dm = run_pyscf(atoms, inp.qm.qmbasis, inp.qm.functional)
        np.save(os.path.join(dirpath, f"dm_conf{geom_idx+1}.npy"), dm)
    end_time = time.time()
    print(f"Calculation finished, wall time cost on DFT: {end_time - start_time:.2f}s")


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

    main(parse_index_str(args.idx), args.cpu)
