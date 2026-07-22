"""Serial-vs-MPI equivalence test of the training pipeline.

SALTED's MPI-parallel code paths differ structurally from the serial ones
(job partitioning via ``distribute_jobs``, hand-sliced ``allreduce``
accumulation of the regression matrices in ``hessian_matrix``). Bugs there
corrupt results might silently rather than crash, so a cheap check is:
run the same small example twice, once serial and once with ``mpirun``, and
require quantitatively similar regression matrices, weights, and validation RMSE.

The prediction surfaces' MPI equivalence lives in ``test_prediction.py``.

Uses the aims water-monomer example (smallest dataset). Carries only the
``mpi`` marker (not ``aims``) so ``-m aims`` and ``-m mpi`` select disjoint sets.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.example, pytest.mark.mpi]

MPI_STEPS = ("sparse_descriptor", "rkhs_vector", "hessian_matrix", "validation")

EXAMPLE = "water_monomer_aims"


@pytest.fixture(scope="module")
def parallel_run(request, make_workspace, mpirun_cmd, serial_run, pipeline_runner):
    # ensure the serial reference exists first (also warms numba caches)
    serial_ws = serial_run(EXAMPLE)
    np_tasks = request.config.getoption("--mpi-np")
    ws = make_workspace(EXAMPLE)
    pipeline_runner(ws, mpi=mpirun_cmd + ["-n", str(np_tasks)], mpi_steps=MPI_STEPS)
    print("\n" + ws.timing_report())
    return serial_ws, ws


def _load(ws, relpath):
    return np.load(ws.root / relpath)


def test_regression_matrices_match(parallel_run):
    serial, par = parallel_run
    rel = f"regrdir_{serial.saltedname}/{serial.mz}"
    for fname in (f"Avec_N{serial.ntrain}.npy", f"Bmat_N{serial.ntrain}.npy"):
        a = _load(serial, f"{rel}/{fname}")
        b = _load(par, f"{rel}/{fname}")
        assert a.shape == b.shape
        # summation order differs across ranks -> allow tiny float noise only
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-10, err_msg=fname)


def test_weights_match(parallel_run):
    serial, par = parallel_run
    rel = f"regrdir_{serial.saltedname}/{serial.mz}/weights_N{serial.ntrain}_{serial.reg_str}.npy"
    np.testing.assert_allclose(_load(serial, rel), _load(par, rel), rtol=1e-6, atol=1e-8)


def test_validation_rmse_matches(parallel_run):
    serial, par = parallel_run
    assert serial.rmse is not None and par.rmse is not None
    assert par.rmse == pytest.approx(serial.rmse, rel=1e-6)
