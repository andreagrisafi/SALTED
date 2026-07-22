"""Tests of SALTED's two prediction surfaces, on the serial run's trained model.

Live-prediction API (``salted.salted_prediction``), called in-process via
``init_pred`` + ``salted_prediction``:

- consistency: predicting a validation structure reproduces the coefficients
  that ``salted.validation`` wrote to ``COEFFS-<iconf+1>.dat``;
- gradients: the analytical coefficient gradients (``gradient=True``) matches
  the central finite differences when the finite-difference displacement is
  small enough, and the finite-difference error converges quadratically with
  the displacement, as expected for a correct gradient.

Prediction pipeline step (``salted.prediction``), run as a subprocess on a
prediction xyz built from the validation structures:

- consistency: the step's ``COEFFS-<k+1>.dat`` (numbered by position in the
  prediction xyz) reproduce the validation step's ``COEFFS-<v[k]+1>.dat``
  (numbered by position in the full dataset).

Serial-vs-MPI equivalence of both surfaces (``mpi``-marked, aims dataset):
``salted.prediction`` distributes the prediction structures across ranks;
``salted.salted_prediction`` distributes the atoms of one structure and
allreduces the partial coefficients (via ``_mpi_predict_driver.py``).
"""

import re
from pathlib import Path

import numpy as np
import pytest
from ase.io import read

pytestmark = pytest.mark.example

EXAMPLE_PARAMS = [
    pytest.param("water_monomer_aims", marks=pytest.mark.aims),
    pytest.param("water_monomer_pyscf", marks=pytest.mark.pyscf),
    pytest.param("water_monomer_cp2k", marks=pytest.mark.cp2k),
]


@pytest.fixture
def load_predictor(serial_run, monkeypatch):
    """Load the trained model of an example's workspace; return (ws, predict).

    ``predict(structure, gradient=False)`` returns the ``salted_prediction``
    output list: ``[coefs]``/``[coefs, grad]`` (+ charge, dipole).
    Chdir into the workspace: SALTED reads ./inp.yaml from the cwd.
    """

    def _load(example: str):
        ws = serial_run(example)
        monkeypatch.chdir(ws.root)
        from salted import init_pred, salted_prediction
        from salted.sys_utils import detect_mpi

        comm, size, rank, _ = detect_mpi()
        model = init_pred.build(rank)
        # no angular truncation: predict every channel of the density-fitting
        # basis (lcut caps the output lambda channels at min(lmax, lcut))
        lcut = model[2]  # lmax_max

        def predict(structure, gradient=False):
            return salted_prediction.build(
                *model, comm, size, rank, lcut, gradient, structure
            )

        return ws, predict

    return _load


@pytest.mark.parametrize("example", EXAMPLE_PARAMS)
def test_prediction_matches_validation(load_predictor, example):
    """Live prediction must reproduce the validation step's coefficients."""
    ws, predict = load_predictor(example)
    frames = read(ws.root / ws.inp["system"]["filename"], ":")

    vdir = ws.validation_output_dir
    # COEFFS-<n>.dat is named by the 1-based configuration index (works for
    # both sequential and random training-set selection)
    coeff_files = sorted(
        vdir.glob("COEFFS-*.dat"),
        key=lambda f: int(re.fullmatch(r"COEFFS-(\d+)\.dat", f.name).group(1)),
    )
    assert coeff_files, f"no COEFFS-*.dat files in {vdir}"

    # all validation configurations must match; collect every deviation
    failures = []
    for fpath in coeff_files:
        iconf = int(re.fullmatch(r"COEFFS-(\d+)\.dat", fpath.name).group(1)) - 1
        coefs = predict(frames[iconf].copy())[0]
        ref = np.loadtxt(fpath)
        assert coefs.shape == ref.shape
        if not np.allclose(coefs, ref):
            abs_err = np.max(np.abs(coefs - ref))
            rel_err = np.linalg.norm(coefs - ref) / np.linalg.norm(ref)
            failures.append(
                f"  conf {iconf} ({fpath.name}): max|diff| = {abs_err:.3e}, "
                f"relative = {rel_err:.3e}"
            )
    assert not failures, (
        f"{example}: prediction deviates from the validation output for "
        f"{len(failures)}/{len(coeff_files)} configurations:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("example", EXAMPLE_PARAMS)
def test_prediction_gradient_finite_difference(load_predictor, example):
    """Analytical gradients must agree with second-order finite differences."""
    ws, predict = load_predictor(example)
    structure = read(ws.root / ws.inp["system"]["filename"], ":")[-1]
    iat, axis = 0, 1  # displace the zeroth atom along y

    g_ana = predict(structure.copy(), gradient=True)[1][iat, axis, :]
    g_norm = np.linalg.norm(g_ana)
    assert np.all(np.isfinite(g_ana)) and g_norm > 0

    err = {}
    for d in (0.01, 0.001):
        plus = structure.copy()
        plus.positions[iat, axis] += d
        minus = structure.copy()
        minus.positions[iat, axis] -= d
        g_fd = (predict(plus)[0] - predict(minus)[0]) / (2 * d)
        err[d] = np.linalg.norm(g_fd - g_ana)

    print(
        f"\n{example}: |g_ana| = {g_norm:.3e}, "
        f"FD error {err[0.01]:.3e} (d=0.01) -> {err[0.001]:.3e} (d=0.001), "
        f"ratio {err[0.01] / err[0.001]:.1f} (ideal 2nd order: 100)"
    )
    assert err[0.001] < err[0.01] / 30, (
        f"{example}: finite-difference error does not converge at 2nd order "
        f"(ratio {err[0.01] / err[0.001]:.1f}, expected ~100): the analytical "
        "gradient disagrees with the predicted coefficients"
    )
    assert err[0.001] < 1e-3 * g_norm, (
        f"{example}: gradient relative error {err[0.001] / g_norm:.3e} at "
        "d=0.001 exceeds 1e-3"
    )


VALSET_PREDNAME = "valset"
# the MPI variants below run on the smallest dataset only; they carry only the
# ``mpi`` marker (not ``aims``) so ``-m aims`` and ``-m mpi`` select disjoint sets
MPI_EXAMPLE = "water_monomer_aims"


@pytest.fixture(scope="module")
def valset_prediction(serial_run):
    """Run ``salted.prediction`` on an example's validation structures, once.

    The validation structures are sliced out of the dataset xyz (using the
    training-set file the pipeline wrote) into a prediction xyz, inp.yaml is
    temporarily retargeted at it (ParseConfig hard-codes the inp.yaml name;
    a distinct predname keeps the outputs in their own directory), and the
    step runs on the already-trained model — no retraining. Cached so the
    serial run is shared between the consistency test and the MPI test.
    """
    cache = {}

    def _get(example: str):
        if example not in cache:
            ws = serial_run(example)
            xyz_fpath = ws.write_prediction_from_validation()
            with ws.swap_prediction_inp(f"./{xyz_fpath.name}", VALSET_PREDNAME):
                ws.run_step("prediction")
            cache[example] = ws
        return cache[example]

    return _get


@pytest.mark.parametrize("example", EXAMPLE_PARAMS)
def test_prediction_step_matches_validation(valset_prediction, example):
    """``salted.prediction`` on the validation structures must reproduce the
    validation step's coefficients."""
    ws = valset_prediction(example)
    vidx = ws.validation_indices()
    assert len(vidx) == ws.nconf - ws.ntrain

    vdir = ws.validation_output_dir
    # guard: the validation step must have predicted exactly these structures,
    # else the training-set selection logic changed and the mapping below is wrong
    written = {int(f.stem.split("-")[1]) - 1 for f in vdir.glob("COEFFS-*.dat")}
    assert written == set(vidx.tolist()), (
        f"{example}: validation COEFFS indices disagree with the training-set "
        f"complement (train/validation split logic changed?)"
    )

    pdir = ws.prediction_output_dir(VALSET_PREDNAME)
    failures = []
    for k, iconf in enumerate(vidx):
        pred = np.loadtxt(pdir / f"COEFFS-{k + 1}.dat")
        ref = np.loadtxt(vdir / f"COEFFS-{iconf + 1}.dat")
        assert pred.shape == ref.shape
        if not np.allclose(pred, ref):
            abs_err = np.max(np.abs(pred - ref))
            rel_err = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
            failures.append(
                f"  conf {iconf} (COEFFS-{k + 1}.dat vs COEFFS-{iconf + 1}.dat): "
                f"max|diff| = {abs_err:.3e}, relative = {rel_err:.3e}"
            )
    assert not failures, (
        f"{example}: salted.prediction deviates from the validation output for "
        f"{len(failures)}/{len(vidx)} configurations:\n" + "\n".join(failures)
    )


@pytest.mark.mpi
def test_prediction_step_mpi_matches_serial(request, valset_prediction, mpirun_cmd):
    """``salted.prediction`` splits the prediction structures across ranks;
    an MPI rerun must agree with the cached serial valset run."""
    ws = valset_prediction(MPI_EXAMPLE)
    xyz_fpath = ws.write_prediction_from_validation()  # deterministic rewrite
    np_tasks = request.config.getoption("--mpi-np")

    with ws.swap_prediction_inp(f"./{xyz_fpath.name}", "valmpi"):
        ws.run_step("prediction", mpi=mpirun_cmd + ["-n", str(np_tasks)])

    sdir = ws.prediction_output_dir(VALSET_PREDNAME)
    mdir = ws.prediction_output_dir("valmpi")
    for k in range(1, len(ws.validation_indices()) + 1):
        a = np.loadtxt(sdir / f"COEFFS-{k}.dat")
        b = np.loadtxt(mdir / f"COEFFS-{k}.dat")
        # each structure is computed entirely by one rank -> tiny noise at most
        np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10, err_msg=f"COEFFS-{k}.dat")


@pytest.mark.mpi
def test_salted_prediction_mpi_matches_serial(request, serial_run, mpirun_cmd):
    """``salted.salted_prediction`` splits the atoms of one structure across
    ranks and allreduces the partial coefficients; serial and MPI must agree.

    Run through ``_mpi_predict_driver.py``: an in-process call cannot be
    mpirun'd, so both runs go through the same driver subprocess.
    """
    ws = serial_run(MPI_EXAMPLE)
    driver = Path(__file__).with_name("_mpi_predict_driver.py")
    xyz = ws.inp["system"]["filename"]  # driver predicts its first structure
    # the atom-parallel path caps the rank count at natoms
    np_tasks = min(request.config.getoption("--mpi-np"), 2)

    ws.run_python([str(driver), xyz, "pred_api_serial.npy"], label="salted_prediction")
    ws.run_python(
        [str(driver), xyz, "pred_api_mpi.npy"],
        label="salted_prediction",
        mpi=mpirun_cmd + ["-n", str(np_tasks)],
    )

    a = np.load(ws.root / "pred_api_serial.npy")
    b = np.load(ws.root / "pred_api_mpi.npy")
    np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)
