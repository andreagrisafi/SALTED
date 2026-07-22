"""Serial-vs-MPI equivalence tests.

SALTED's MPI-parallel code paths differ structurally from the serial ones
(job partitioning via ``distribute_jobs``, hand-sliced ``allreduce``
accumulation of the regression matrices in ``hessian_matrix``). Bugs there
corrupt results might silently rather than crash, so a cheap check is:
run the same small example twice, once serial and once with ``mpirun``, and
require quantitatively similar regression matrices, weights, and validation RMSE.

Both prediction surfaces are covered too, each with its own MPI strategy:

- ``salted.prediction`` (pipeline step) distributes the *structures* of the
  prediction set across ranks;
- ``salted.salted_prediction`` (live API) distributes the *atoms* of one structure
  across ranks and ``allreduce``s the partial coefficients (run through
  ``_mpi_predict_driver.py``, since an in-process call cannot be mpirun'd).

Uses the aims water-monomer example (smallest dataset).
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.example, pytest.mark.mpi, pytest.mark.aims]

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


def test_prediction_step_mpi_matches_serial(request, serial_run, mpirun_cmd):
    """``salted.prediction`` splits the prediction structures across ranks;
    serial and MPI runs on the same trained model must agree.

    Predicts the validation structures (point-1 mechanics: slice the dataset
    xyz by the training-set complement, temporarily retarget inp.prediction).
    Distinct prednames keep the two runs' output directories apart.
    """
    ws = serial_run(EXAMPLE)
    xyz = ws.write_prediction_from_validation()
    np_tasks = request.config.getoption("--mpi-np")

    with ws.swap_prediction_inp(f"./{xyz.name}", "valser"):
        ws.run_step("prediction")
    with ws.swap_prediction_inp(f"./{xyz.name}", "valmpi"):
        ws.run_step("prediction", mpi=mpirun_cmd + ["-n", str(np_tasks)])

    sdir = ws.prediction_output_dir("valser")
    mdir = ws.prediction_output_dir("valmpi")
    for k in range(1, len(ws.validation_indices()) + 1):
        a = np.loadtxt(sdir / f"COEFFS-{k}.dat")
        b = np.loadtxt(mdir / f"COEFFS-{k}.dat")
        # each structure is computed entirely by one rank -> tiny noise at most
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10, err_msg=f"COEFFS-{k}.dat")


def test_salted_prediction_mpi_matches_serial(request, serial_run, mpirun_cmd):
    """``salted_prediction`` splits the atoms of one structure across ranks
    and allreduces the partial coefficients; serial and MPI must agree."""
    ws = serial_run(EXAMPLE)
    driver = Path(__file__).with_name("_mpi_predict_driver.py")
    xyz = ws.inp["system"]["filename"]  # driver predicts its first structure
    # the atom-parallel path caps the rank count at natoms (3 for a water monomer)
    np_tasks = min(request.config.getoption("--mpi-np"), 3)

    ws.run_python([str(driver), xyz, "pred_api_serial.npy"], label="salted_prediction")
    ws.run_python(
        [str(driver), xyz, "pred_api_mpi.npy"],
        label="salted_prediction",
        mpi=mpirun_cmd + ["-n", str(np_tasks)],
    )

    a = np.load(ws.root / "pred_api_serial.npy")
    b = np.load(ws.root / "pred_api_mpi.npy")
    # allreduce summation order differs across ranks -> allow tiny float noise
    np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)
