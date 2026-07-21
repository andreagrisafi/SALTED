"""End-to-end tests of the SALTED ML pipeline on the example datasets.

These reproduce the SALTED workflow using the precomputed density-fitting data
from the SALTED-datasets repository:

    initialize -> sparse_selection -> sparse_descriptor -> rkhs_projector
    -> rkhs_vector -> hessian_matrix -> solve_regression -> validation

Each step's expected artifacts are checked, and the final density RMSE
reported by ``salted.validation`` must stay below a calibrated threshold.

Run with ``pytest -m example`` (needs --datasets-path / $SALTED_DATASETS_PATH).
Reduce the training-set size for a faster run with ``--ntrain N``.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.example


def check_artifacts(ws):
    """Assert every pipeline step left its expected artifacts behind."""
    root = ws.root
    name = ws.saltedname
    sdir = root / f"equirepr_{name}"

    # initialize: wigner symbols + scalar (lambda=0) descriptor
    assert list((root / "wigners").glob("wigner_lam-*.dat")), "no wigner files"
    assert (sdir / "FEAT-0.h5").is_file()
    ncut = ws.inp["descriptor"].get("sparsify", {}).get("ncut", 0)
    if ncut > 0:
        assert list(sdir.glob(f"fps{ncut}-*.npy")), "no feature-sparsification files"

    # sparse_selection: FPS-selected environment indices
    menv = ws.inp["gpr"]["Menv"]
    sparse_set = sdir / f"sparse_set_{menv}.txt"
    assert sparse_set.is_file()
    idx = np.loadtxt(sparse_set, dtype=int)
    assert idx.shape[0] == menv

    # sparse_descriptor + rkhs_projector
    assert (sdir / f"FEAT_M-{menv}.h5").is_file()
    zeta = ws.inp["gpr"]["z"]
    assert (sdir / f"projector_M{menv}_zeta{zeta}.h5").is_file()

    # rkhs_vector: one sparse feature vector per configuration
    psi_dir = root / f"rkhs-vectors_{name}" / ws.mz
    psi_files = list(psi_dir.glob("psi-nm_conf*.npz"))
    assert len(psi_files) == ws.nconf, (
        f"expected {ws.nconf} psi-nm vectors, found {len(psi_files)}"
    )

    # hessian_matrix + solve_regression
    rdir = root / f"regrdir_{name}" / ws.mz
    avec = rdir / f"Avec_N{ws.ntrain}.npy"
    bmat = rdir / f"Bmat_N{ws.ntrain}.npy"
    weights = rdir / f"weights_N{ws.ntrain}_{ws.reg_str}.npy"
    for f in (avec, bmat, weights):
        assert f.is_file(), f"missing {f}"
        arr = np.load(f)
        assert np.all(np.isfinite(arr)), f"non-finite values in {f}"

    # validation: per-configuration error file
    vdir = root / f"validations_{name}" / ws.mz / f"N{ws.ntrain}_{ws.reg_str}"
    errors = np.loadtxt(vdir / "errors.dat")
    n_val = ws.nconf - ws.ntrain
    assert errors.shape[0] == n_val, (
        f"expected {n_val} validation entries, found {errors.shape[0]}"
    )


@pytest.mark.parametrize(
    "example",
    [
        pytest.param("water_monomer_aims", marks=pytest.mark.aims),
        pytest.param("water_monomer_pyscf", marks=pytest.mark.pyscf),
        pytest.param("water_monomer_cp2k", marks=pytest.mark.cp2k),
    ],
)
def test_example_pipeline(serial_run, example):
    ws = serial_run(example)
    check_artifacts(ws)
    assert ws.rmse is not None
    assert ws.rmse < ws.rmse_threshold, (
        f"{example}: validation RMSE {ws.rmse:.3e}% exceeds threshold "
        f"{ws.rmse_threshold}%"
    )
