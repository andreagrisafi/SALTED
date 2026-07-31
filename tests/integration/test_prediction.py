"""SALTED predicts through two surfaces:``salted.salted_prediction``
and ``salted.prediction``.
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from conftest import PipelineWorkspace, WorkspaceBuilder
from models import ALL_MODELS, LIVE_API_MODELS, LIVE_API_MPI_MODELS, MPI_MODELS

pytestmark = pytest.mark.integration

VALSET_PREDNAME = "valset"

# predict(structure, gradient=False) -> [coefs] or [coefs, grad]; Callable cannot
# express the default argument, hence the ellipsis
Predictor = Callable[..., list[np.ndarray]]


@pytest.fixture
def predictor(serial_run: PipelineWorkspace, monkeypatch) -> Predictor:
    """Returns ``predict(structure, gradient=False)``, giving the
    ``salted_prediction`` output list ``[coefs]`` or ``[coefs, grad]``.

    Chdirs into the workspace because SALTED reads ./inp.yaml from the cwd.
    """
    monkeypatch.chdir(serial_run.root)
    from salted import init_pred, salted_prediction
    from salted.sys_utils import detect_mpi

    comm, size, rank, _ = detect_mpi()
    model = init_pred.build(rank)
    lcut = model[2]  # lmax_max

    def predict(structure: Atoms, gradient: bool = False) -> list[np.ndarray]:
        return salted_prediction.build(*model, comm, size, rank, lcut, gradient, structure)

    return predict


@pytest.fixture(scope="session")
def valset_prediction(serial_run: PipelineWorkspace) -> Path:
    """Run ``salted.prediction`` on the model's validation structures, once.
    Returns the directory it wrote coefficients to.

    ParseConfig hard-codes ``<cwd>/inp.yaml``, so the prediction target cannot
    be passed in; the file itself is swapped and restored around the run.
    """
    xyz_fpath = serial_run.write_prediction_from_validation()
    with serial_run.swap_prediction_inp(f"./{xyz_fpath.name}", VALSET_PREDNAME):
        serial_run.run_step("prediction")
    return serial_run.prediction_output_dir(VALSET_PREDNAME)


@pytest.mark.parametrize("model_spec", LIVE_API_MODELS, indirect=True)
def test_prediction_matches_validation(serial_run: PipelineWorkspace, predictor: Predictor):
    """Live prediction must reproduce the validation step's coefficients."""
    frames = read(serial_run.root / serial_run.inp["system"]["filename"], ":")

    vdir = serial_run.validation_output_dir
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
        coefs = predictor(frames[iconf].copy())[0]
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
        f"{serial_run.name}: prediction deviates from the validation output for "
        f"{len(failures)}/{len(coeff_files)} configurations:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("model_spec", LIVE_API_MODELS, indirect=True)
def test_prediction_gradient_finite_difference(serial_run: PipelineWorkspace, predictor: Predictor):
    """Analytical gradients must agree with second-order finite differences."""
    structure = read(serial_run.root / serial_run.inp["system"]["filename"], ":")[-1]
    iat, axis = 0, 1  # displace the zeroth atom along y

    g_ana = predictor(structure.copy(), gradient=True)[1][iat, axis, :]
    g_norm = np.linalg.norm(g_ana)
    assert np.all(np.isfinite(g_ana)) and g_norm > 0

    err = {}
    for d in (0.01, 0.001):
        plus = structure.copy()
        plus.positions[iat, axis] += d
        minus = structure.copy()
        minus.positions[iat, axis] -= d
        g_fd = (predictor(plus)[0] - predictor(minus)[0]) / (2 * d)
        err[d] = np.linalg.norm(g_fd - g_ana)

    print(
        f"\n{serial_run.name}: |g_ana| = {g_norm:.3e}, "
        f"FD error {err[0.01]:.3e} (d=0.01) -> {err[0.001]:.3e} (d=0.001), "
        f"ratio {err[0.01] / err[0.001]:.1f} (ideal 2nd order: 100)"
    )
    assert err[0.001] < err[0.01] / 30, (
        f"{serial_run.name}: finite-difference error does not converge at 2nd order "
        f"(ratio {err[0.01] / err[0.001]:.1f}, expected ~100): the analytical "
        "gradient disagrees with the predicted coefficients"
    )
    assert err[0.001] < 1e-3 * g_norm, (
        f"{serial_run.name}: gradient relative error {err[0.001] / g_norm:.3e} at "
        "d=0.001 exceeds 1e-3"
    )


@pytest.mark.parametrize("model_spec", ALL_MODELS, indirect=True)
def test_prediction_step_matches_validation(serial_run: PipelineWorkspace, valset_prediction: Path):
    """Every COEFFS file, and for ``density-response`` every Cartesian
    component, must match what the validation step wrote."""
    ws = serial_run
    vidx = ws.validation_indices()
    assert len(vidx) == ws.nconf - ws.ntrain

    vdir = ws.validation_output_dir
    # the validation step must have predicted exactly these structures,
    # else the training-set selection logic changed and the mapping below is wrong
    written = {int(f.stem.split("-")[1]) - 1 for f in ws.coeff_dirs(vdir)[0].glob("COEFFS-*.dat")}
    assert written == set(vidx.tolist()), (
        f"{ws.name}: validation COEFFS indices disagree with the training-set "
        f"complement (train/validation split logic changed?)"
    )

    failures = []
    n_checked = 0
    for k, iconf in enumerate(vidx):
        # one file per structure for density, one per Cartesian component for
        # density-response; both lists come back in the same component order
        for pred_f, ref_f in zip(
            ws.coeff_files(valset_prediction, k + 1), ws.coeff_files(vdir, iconf + 1)
        ):
            pred = np.loadtxt(pred_f)
            ref = np.loadtxt(ref_f)
            assert pred.shape == ref.shape
            n_checked += 1
            if not np.allclose(pred, ref):
                abs_err = np.max(np.abs(pred - ref))
                rel_err = np.linalg.norm(pred - ref) / np.linalg.norm(ref)
                failures.append(
                    f"  conf {iconf} ({pred_f.relative_to(valset_prediction)} vs "
                    f"{ref_f.relative_to(vdir)}): "
                    f"max|diff| = {abs_err:.3e}, relative = {rel_err:.3e}"
                )
    assert not failures, (
        f"{ws.name}: salted.prediction deviates from the validation output for "
        f"{len(failures)}/{n_checked} coefficient files:\n" + "\n".join(failures)
    )


@pytest.mark.parametrize("model_spec", MPI_MODELS, indirect=True)
def test_prediction_step_mpi_matches_serial(
    serial_run: PipelineWorkspace, valset_prediction: Path, workspaces: WorkspaceBuilder
):
    """``salted.prediction`` splits the prediction structures across ranks;
    an MPI rerun must agree with the cached serial valset run."""
    ws = serial_run
    xyz_fpath = ws.write_prediction_from_validation()  # deterministic rewrite

    with ws.swap_prediction_inp(f"./{xyz_fpath.name}", "valmpi"):
        ws.run_step("prediction", mpi=workspaces.mpi_cmd)

    mdir = ws.prediction_output_dir("valmpi")
    for k in range(1, len(ws.validation_indices()) + 1):
        # one file per structure for density, one per Cartesian component for
        # density-response; both lists come back in the same component order
        for serial_f, mpi_f in zip(ws.coeff_files(valset_prediction, k), ws.coeff_files(mdir, k)):
            a = np.loadtxt(serial_f)
            b = np.loadtxt(mpi_f)
            # each structure is computed entirely by one rank -> tiny noise at most
            np.testing.assert_allclose(
                b, a, rtol=1e-8, atol=1e-10, err_msg=str(mpi_f.relative_to(mdir))
            )


@pytest.mark.parametrize("model_spec", LIVE_API_MPI_MODELS, indirect=True)
def test_salted_prediction_mpi_matches_serial(
    serial_run: PipelineWorkspace, workspaces: WorkspaceBuilder
):
    """An in-process call cannot be mpirun'd, so both runs go through the same
    ``_mpi_predict_driver.py`` subprocess. One model suffices: this exercises
    SALTED's atom partitioning, not anything dataset-specific.
    """
    ws = serial_run
    driver = Path(__file__).with_name("_mpi_predict_driver.py")
    xyz = ws.inp["system"]["filename"]  # driver predicts its first structure
    # the atom-parallel path caps the rank count at natoms
    np_tasks = min(workspaces.np_tasks, 2)

    ws.run_python([str(driver), xyz, "pred_api_serial.npy"], label="salted_prediction")
    ws.run_python(
        [str(driver), xyz, "pred_api_mpi.npy"],
        label="salted_prediction",
        mpi=workspaces.mpirun + ["-n", str(np_tasks)],
    )

    a = np.load(ws.root / "pred_api_serial.npy")
    b = np.load(ws.root / "pred_api_mpi.npy")
    np.testing.assert_allclose(b, a, rtol=1e-8, atol=1e-10)
