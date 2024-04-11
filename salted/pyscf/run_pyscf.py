import argparse
import os
import sys
from typing import List, Tuple, Union
import time

import numpy as np
from ase.io import read
from pyscf import gto, dft, lib

from salted.sys_utils import ParseConfig



def run_pyscf(
    atoms: List,
    basis: str,
    xc: str,
):
    mol = gto.M(atom=atoms, basis=basis, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    mf.kernel()
    return mf.make_rdm1()


def main(geom_indexes: Union[List[int], None], num_threads: int):
    inp = ParseConfig().parse_input()
    geoms_all = read(inp.system.filename, ":")
    if geom_indexes is None:
        geom_indexes = range(len(geoms_all))
    else:
        geom_indexes = [i for i in geom_indexes if i < len(geoms_all)]  # indexes start from 0
    print(f"Calculating density matrix for configurations: {geom_indexes}")
    geoms = [geoms_all[i] for i in geom_indexes]

    """ prepare the output directory """
    dirpath = os.path.join(inp.qm.path2qm, "density_matrices")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    """ set pyscf.lib.num_threads """
    lib.num_threads(num_threads)

    """ do DFT calculation """
    start_time = time.time()
    for cal_idx, (geom_idx, geom) in enumerate(zip(geom_indexes, geoms)):
        print(f"calcualte {geom_idx=}, progress: {cal_idx}/{len(geom_indexes)}")
        symb = geom.get_chemical_symbols()
        coords = geom.get_positions()
        atoms = [(s, c) for s, c in zip(symb, coords)]

        dm = run_pyscf(atoms, inp.qm.qmbasis, inp.qm.functional)
        np.save(os.path.join(dirpath, f"dm_conf{geom_idx}.npy"), dm)
    end_time = time.time()
    print(f"Calculation finished, time cost on DFT: {end_time - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # create a parser obj, which accepts the indexes to calculate, start from 0
    # formats: 1,2,3 or 1-3 or None (all structures)
    parser.add_argument(
        "-i", "--idx", type=str, default="all",
        help="Indexes to calculate, start from 0. Format: 1,3-5,7-10. Default is \"all\", which means all structures."
    )
    parser.add_argument(
        "-c", "--cpu", type=int, default=1,
        help="Number of CPU cores to use. Default is 1."
    )
    args = parser.parse_args()

    if args.idx == "all":
        indexes = None
    else:
        assert isinstance(args.idx, str)
        indexes = []
        for s in args.idx.split(","):  # e.g. ["1", "3-5", "7-10"]
            assert all([c.isdigit() or c == "-" for c in s]), f"Invalid index format: {s}"
            if "-" in s:
                assert s.count("-") == 1, f"Invalid index format: {s}"
                start, end = map(int, s.split("-"))
                indexes.extend(range(start, end + 1))
            else:
                indexes.append(int(s))

    main(indexes, args.cpu)
