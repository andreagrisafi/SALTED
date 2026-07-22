"""MPI driver for the live-prediction API, used by test_mpi_equivalence.

``salted_prediction`` parallelises over the atoms of a single structure and
``allreduce``s the partial coefficient vectors, so its MPI path can only be
exercised in a real ``mpirun`` process — hence this driver instead of an
in-process pytest call. Run it inside a trained SALTED workspace (the cwd
must contain inp.yaml):

    [mpirun -n N] python _mpi_predict_driver.py <structure.xyz> <out.npy>

Predicts the first structure of <structure.xyz> and saves the coefficient
vector to <out.npy> (rank 0 only). N must not exceed the atom count.
"""

import sys

import numpy as np
from ase.io import read

from salted import init_pred, salted_prediction
from salted.sys_utils import detect_mpi


def main():
    xyz_fpath, out_fpath = sys.argv[1], sys.argv[2]
    comm, size, rank, _ = detect_mpi()
    model = init_pred.build(rank)
    # no angular truncation: predict every channel of the density-fitting
    # basis (lcut caps the output lambda channels at min(lmax, lcut))
    lcut = model[2]  # lmax_max
    structure = read(xyz_fpath, 0)  # explicit: the first frame (ASE defaults to the last)
    out = salted_prediction.build(*model, comm, size, rank, lcut, False, structure)
    if rank == 0:
        np.save(out_fpath, out[0])


if __name__ == "__main__":
    main()
