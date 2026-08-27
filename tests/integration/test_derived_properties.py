"""Some properties can be derived from the predicted density coefficients.
Now this file is for cp2k_density models only. See DERIVED_PROPERTY_MODELS.
"""

import numpy as np
import pytest
from ase.io import read
from conftest import PipelineWorkspace
from models import DERIVED_PROPERTY_MODELS

pytestmark = pytest.mark.integration


@pytest.fixture
def tols(serial_run: PipelineWorkspace) -> dict[str, float]:
    """Calibrated tolerances; x10 under a --ntrain reduced run, mirroring the
    loose RMSE bound the pipeline test switches to."""
    tols = serial_run.spec.derived_property_tols
    assert tols, "derived-property models must set ModelSpec.derived_property_tols"
    relax = 10.0 if serial_run.ntrain_reduced else 1.0
    return {name: tol * relax for name, tol in tols.items()}


@pytest.fixture
def load_property(serial_run: PipelineWorkspace):
    """Load a property file, checking it covers exactly the validation set."""
    vidx = serial_run.validation_indices()

    def load(name: str, ncols: int) -> np.ndarray:
        rows = serial_run.validation_property(name)
        assert rows.shape == (len(vidx), ncols), (
            f"{name}.dat: expected one row per validation structure "
            f"({len(vidx)} x {ncols}), got {rows.shape}"
        )
        assert np.array_equal(np.sort(rows[:, 0].astype(int)), vidx + 1)
        return rows

    return load


@pytest.mark.parametrize("model_spec", DERIVED_PROPERTY_MODELS, indirect=True)
def test_total_charge(serial_run: PipelineWorkspace, load_property, tols):
    """Predicted vs reference total charge, and reference vs valence charge:
    the zeroth moment of the fitted density must equal the sum of the
    species' pseudo charges."""
    ws = serial_run
    rows = load_property("charges", 3)

    frames = read(ws.root / ws.inp["system"]["filename"], ":")
    pseudocharge = {
        spe: np.loadtxt(ws.root / "basis" / f"{spe}-local_pseudo.dat")[0]
        for spe in ws.inp["system"]["species"]
    }
    valence = np.array(
        [sum(pseudocharge[s] for s in frames[int(i) - 1].get_chemical_symbols()) for i in rows[:, 0]]
    )

    ref_err = np.abs(rows[:, 1] - valence)
    assert ref_err.max() < tols["charge"], (
        f"{ws.name}: reference total charge deviates from the valence charge "
        f"by up to {ref_err.max():.3e} e (wrong pseudo charges or overlaps?)"
    )
    pred_err = np.abs(rows[:, 2] - rows[:, 1])
    assert pred_err.max() < tols["charge"], (
        f"{ws.name}: predicted total charge deviates from reference by up to "
        f"{pred_err.max():.3e} e (tolerance {tols['charge']:.1e})"
    )


@pytest.mark.parametrize("model_spec", DERIVED_PROPERTY_MODELS, indirect=True)
def test_dipole_moment(serial_run: PipelineWorkspace, load_property, tols):
    """Predicted vs reference dipole moment, per Cartesian component."""
    rows = load_property("dipoles", 7)
    err = np.abs(rows[:, 4:7] - rows[:, 1:4])
    assert err.max() < tols["dipole"], (
        f"{serial_run.name}: predicted dipole deviates from reference by up to "
        f"{err.max():.3e} a.u. (tolerance {tols['dipole']:.1e})"
    )


@pytest.mark.parametrize("model_spec", DERIVED_PROPERTY_MODELS, indirect=True)
def test_hartree_energy(serial_run: PipelineWorkspace, load_property, tols):
    """Predicted vs reference Hartree energy, as a relative error: the
    absolute energy (~18.5 Ha for a water monomer) is dominated by terms any
    density of the right total charge reproduces."""
    ws = serial_run
    if ws.inp["qm"].get("dfmetric") != "coulomb":
        pytest.skip("Hartree energy is only computed with qm.dfmetric: coulomb")
    rows = load_property("electrostatic_energy", 3)
    assert np.all(np.isfinite(rows)) and np.all(np.abs(rows[:, 1]) > 1.0)
    rel_err = np.abs(rows[:, 2] - rows[:, 1]) / np.abs(rows[:, 1])
    assert rel_err.max() < tols["hartree"], (
        f"{ws.name}: predicted Hartree energy deviates from reference by up to "
        f"{rel_err.max():.3e} relative (tolerance {tols['hartree']:.1e})"
    )
