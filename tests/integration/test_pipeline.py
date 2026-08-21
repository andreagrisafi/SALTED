"""End-to-end runs of each model's pipeline: every step's artifacts, then the
validation RMSE against the model's calibrated threshold.

``STEP_CHECKS`` is keyed by step name and driven by ``spec.steps``, the same
list the runner uses, so a model that skips a step skips its check too.
"""

from collections.abc import Callable

import numpy as np
import pytest
from conftest import PipelineWorkspace
from models import ALL_MODELS

pytestmark = pytest.mark.integration


def _check_initialize(ws: PipelineWorkspace) -> None:
    assert list((ws.root / "wigners").glob("wigner_lam-*.dat")), "no wigner files"
    if ws.is_response:
        # density-response needs one extra lambda channel and the antisymmetric
        # part of the two-body descriptor, hence a second set of Wigner symbols
        assert list((ws.root / "wigners").glob("wigner_antisymm_lam-*.dat")), (
            "no antisymmetric wigner files"
        )
    assert (ws.equirepr_dir / "FEAT-0.h5").is_file()

    ncut = ws.inp["descriptor"].get("sparsify", {}).get("ncut", 0)
    if ncut > 0:
        assert list(ws.equirepr_dir.glob(f"fps{ncut}-*.npy")), "no feature-sparsification files"


def _check_sparse_selection(ws: PipelineWorkspace) -> None:
    menv = ws.inp["gpr"]["Menv"]
    sparse_set = ws.equirepr_dir / f"sparse_set_{menv}.txt"
    assert sparse_set.is_file()
    idx = np.loadtxt(sparse_set, dtype=int)
    assert idx.shape[0] == menv


def _check_sparse_descriptor(ws: PipelineWorkspace) -> None:
    menv = ws.inp["gpr"]["Menv"]
    assert (ws.equirepr_dir / f"FEAT_M-{menv}.h5").is_file()
    if ws.is_response:
        assert (ws.equirepr_dir / f"FEAT_M-{menv}_antisymm.h5").is_file()


def _check_rkhs_projector(ws: PipelineWorkspace) -> None:
    menv, zeta = ws.inp["gpr"]["Menv"], ws.inp["gpr"]["z"]
    stem = "projector-response" if ws.is_response else "projector"
    assert (ws.equirepr_dir / f"{stem}_M{menv}_zeta{zeta}.h5").is_file()


def _check_rkhs_vector(ws: PipelineWorkspace) -> None:
    # one vector per configuration, and per Cartesian component of dn/dE
    psi_files = list(ws.rkhs_vector_dir.glob("psi-nm_conf*.npz"))
    if ws.is_response:
        n_expected = ws.nconf * len(ws.cart_components)
        for icart in ws.cart_components:
            n_cart = len(list(ws.rkhs_vector_dir.glob(f"psi-nm_conf*_{icart}.npz")))
            assert n_cart == ws.nconf, (
                f"expected {ws.nconf} psi-nm vectors for component {icart}, found {n_cart}"
            )
    else:
        n_expected = ws.nconf
    assert len(psi_files) == n_expected, (
        f"expected {n_expected} psi-nm vectors, found {len(psi_files)}"
    )


def _check_finite(ws: PipelineWorkspace, *names: str) -> None:
    for name in names:
        fpath = ws.regression_path(name)
        assert fpath.is_file(), f"missing {fpath}"
        assert np.all(np.isfinite(np.load(fpath))), f"non-finite values in {fpath}"


def _check_hessian_matrix(ws: PipelineWorkspace) -> None:
    _check_finite(ws, "Avec", "Bmat")


def _check_solve_regression(ws: PipelineWorkspace) -> None:
    _check_finite(ws, "weights")


def _check_validation(ws: PipelineWorkspace) -> None:
    vdir = ws.validation_output_dir
    n_val = ws.nconf - ws.ntrain
    errors = np.loadtxt(vdir / "errors.dat")
    assert errors.shape[0] == n_val, (
        f"expected {n_val} validation entries, found {errors.shape[0]}"
    )
    # one COEFFS directory for density, one per Cartesian component otherwise
    for cdir in ws.coeff_dirs(vdir):
        n_coeffs = len(list(cdir.glob("COEFFS-*.dat")))
        assert n_coeffs == n_val, (
            f"expected {n_val} COEFFS files in {cdir}, found {n_coeffs}"
        )


STEP_CHECKS: dict[str, Callable[[PipelineWorkspace], None]] = {
    "initialize": _check_initialize,
    "sparse_selection": _check_sparse_selection,
    "sparse_descriptor": _check_sparse_descriptor,
    "rkhs_projector": _check_rkhs_projector,
    "rkhs_vector": _check_rkhs_vector,
    "hessian_matrix": _check_hessian_matrix,
    "solve_regression": _check_solve_regression,
    "validation": _check_validation,
}


@pytest.mark.parametrize("model_spec", ALL_MODELS, indirect=True)
def test_pipeline(serial_run: PipelineWorkspace):
    ws = serial_run
    for step in ws.spec.steps:
        assert step in STEP_CHECKS, f"no artifact check registered for step {step!r}"
        STEP_CHECKS[step](ws)

    assert ws.validation_rmse is not None
    assert ws.validation_rmse < ws.validation_rmse_threshold, (
        f"{ws.name}: validation RMSE {ws.validation_rmse:.3e}% exceeds threshold {ws.validation_rmse_threshold}%"
    )
