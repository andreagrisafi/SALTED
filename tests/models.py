"""Registry of the models in integration tests.

A model will be trained on one dataset to verify the end-to-end pipeline, and might be further tested.
To add a new model, add a ``ModelSpec`` here and register its marker in ``pyproject.toml``.

Adding a model means adding one ``ModelSpec`` here, registering its marker in ``pyproject.toml``,
and adding that marker to the CI matrix in ``.github/workflows/ci.yaml``.
No test module needs to change.
"""

from dataclasses import dataclass
from typing import Any

import pytest

# The canonical SALTED training pipeline, in execution order.
# Change ModelSpec.steps if a model needs to skip a step, e.g. minimize_loss instead of inverting the hessian matrix.
TRAINING_PIPELINE = (
    "initialize",
    "sparse_selection",
    "sparse_descriptor",
    "rkhs_projector",
    "rkhs_vector",
    "hessian_matrix",
    "solve_regression",
    "validation",
)


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to train and validate one model.

    Launcher-agnostic on purpose: the same spec is run serially and under
    ``mpirun`` by the equivalence tests, so both runs are guaranteed to be of
    the *same* model. How to execute is a fixture's concern, not a spec's.
    """

    key: str
    """Registry key; also the pytest parameter id."""

    dataset_dir: str
    """Directory name inside the SALTED-datasets checkout."""

    marker: str
    """pytest marker selecting this model; one CI job per marker."""

    nconf: int
    """Number of structures in the dataset."""

    inp: dict[str, Any]
    """Template for the workspace's inp.yaml (deep-copied before use)."""

    validation_rmse_threshold: float
    """Upper bound (%) on the RMSE ``salted.validation`` reports, calibrated
    per dataset. The quantity is the model's target: the density for
    ``density``, dn/dE for ``density-response``."""

    steps: tuple[str, ...] = TRAINING_PIPELINE
    """Pipeline steps to run, in order. Both the runner and the artifact
    checks read this, so skipping a step also skips its artifact check."""

    weights_rtol: float = 1e-6
    """Tolerance of the serial-vs-MPI weight comparison, as a relative L2
    norm. How far the two weight vectors may drift is a property of the
    regression matrix's conditioning, not of the MPI code."""

    check_mpi: bool = False
    """Include this model in the serial-vs-MPI equivalence tests."""

    fd_gradient: bool = True
    """Check the live-API analytical gradient against finite differences."""

    derived_property_tols: dict[str, float] | None = None
    """Check dervied properties from output RI coefficients, see ``test_derived_properties.py``"""

    @property
    def saltedtype(self) -> str:
        return self.inp["salted"].get("saltedtype", "density")

    @property
    def has_derived_properties(self) -> bool:
        """for cp2k total charge, dipole moment, and Hartree energy"""
        return self.inp["qm"]["qmcode"] == "cp2k" and not self.is_response

    @property
    def is_response(self) -> bool:
        return self.saltedtype == "density-response"

    @property
    def cart_components(self) -> tuple[str, ...]:
        """Cartesian components of dn/dE; empty for a plain density target."""
        return ("x", "y", "z") if self.is_response else ()


DESCRIPTOR_TEMPLATE = {
    "rep1": {
        "type": "rho",
        "rcut": 4.0,
        "sig": 0.3,
        "nrad": 8,
        "nang": 6,
        "neighspe": ["H", "O"],
    },
    "rep2": {
        "type": "rho",
        "rcut": 4.0,
        "sig": 0.3,
        "nrad": 8,
        "nang": 6,
        "neighspe": ["H", "O"],
    },
}


_SPECS = [
    ModelSpec(
        key="water_monomer_aims_density",
        dataset_dir="water_monomer_AIMS",
        marker="aims_density",
        nconf=100,
        check_mpi=True,
        inp={
            "salted": {"saltedname": "test", "saltedpath": "./", "verbose": True},
            "system": {"filename": "./water_monomers_100.xyz", "species": ["H", "O"]},
            "qm": {"path2qm": "./", "qmcode": "aims", "dfbasis": "FHI-aims-light"},
            "prediction": {"filename": "./water_dimers_10.xyz", "predname": "prediction"},
            "descriptor": DESCRIPTOR_TEMPLATE,
            "gpr": {
                "z": 2.0,
                "Menv": 100,
                "Ntrain": 40,
                "trainfrac": 1.0,
                "trainsel": "random",
            },
        },
        # andreagrisafi/SALTED-datasets: water_monomer_AIMS/README.md
        # validation RMSE 9.680e-01 %
        validation_rmse_threshold=1.5,
    ),
    ModelSpec(
        key="water_monomer_aims_response",
        dataset_dir="water_monomer_AIMS_response_subset100",
        marker="aims_response",
        nconf=100,
        check_mpi=True,
        inp={
            "salted": {
                "saltedname": "test",
                "saltedpath": "./",
                "saltedtype": "density-response",
                "verbose": True,
            },
            "system": {
                "filename": "./water_monomers_100.xyz",
                "species": ["H", "O"],
                "average": False,
            },
            "qm": {"path2qm": "./", "qmcode": "aims", "dfbasis": "RI-aims"},
            "prediction": {"filename": "./pred_placeholder.xyz", "predname": "prediction"},
            "descriptor": DESCRIPTOR_TEMPLATE,
            "gpr": {
                "z": 2.0,
                "Menv": 100,
                "Ntrain": 40,
                "trainfrac": 1.0,
                "regul": 1e-8,
                "trainsel": "random",
            },
        },
        # SALTED-datasets: water_monomer_AIMS_response_subset100/README.md
        # 2026-07 Ntrain=40/validation=60 on the 100-structure subset: % RMSE 6.736e-01
        validation_rmse_threshold=1.0,
        # Bmat is rank-deficient here (smallest eigenvalue ~ -3e-16), so larger rtol for weights
        weights_rtol=1e-4,
    ),
    ModelSpec(
        key="water_monomer_cp2k_density",
        dataset_dir="water_monomer_CP2K_subset100",
        marker="cp2k_density",
        nconf=100,
        check_mpi=True,
        inp={
            "salted": {"saltedname": "test", "saltedpath": "./", "verbose": True},
            "system": {"filename": "./water_monomers_100.xyz", "species": ["H", "O"]},
            "qm": {
                "path2qm": "./",
                "qmcode": "cp2k",
                "periodic": "3D",
                "dfmetric": "coulomb",
                "dfbasis": "RI-basis",
            },
            "prediction": {"filename": "./water_dimers_10.xyz", "predname": "prediction"},
            "descriptor": {
                "rep1": DESCRIPTOR_TEMPLATE["rep1"],
                "rep2": {**DESCRIPTOR_TEMPLATE["rep2"], "type": "V"},  # potential rep
                "sparsify": {"nsamples": 10, "ncut": 1000},
            },
            "gpr": {
                "z": 1.0,
                "Menv": 50,
                "Ntrain": 40,
                "trainfrac": 1.0,
                "regul": 1e-5,
                "gradtol": 1e-6,
                "trainsel": "random",
            },
        },
        # 2026-08 Ntrain=40/validation=60 on Coulomb metric: % RMSE 1.497e+00
        validation_rmse_threshold=2.5,
        fd_gradient=False,
        # 2026-08 observed maxima on the same run: charge 1.3e-2, dipole 1.9e-2, hartree 1.2e-3
        derived_property_tols={"charge": 0.05, "dipole": 0.05, "hartree": 5e-3},
    ),
    ModelSpec(
        key="water_monomer_pyscf_density",
        dataset_dir="water_monomer_PySCF_subset100",
        marker="pyscf_density",
        nconf=100,
        inp={
            "salted": {"saltedname": "test", "saltedpath": "./", "verbose": True},
            "system": {"filename": "./water_monomers_100.xyz", "species": ["H", "O"]},
            "qm": {
                "path2qm": "./",
                "qmcode": "pyscf",
                "dfbasis": "RI-cc-pvqz",
                "qmbasis": "cc-pvqz",
                "functional": "b3lyp",
            },
            "prediction": {"filename": "./water_dimers_10.xyz", "predname": "dimer"},
            "descriptor": {
                **DESCRIPTOR_TEMPLATE,
                "sparsify": {"nsamples": 100, "ncut": 1000},
            },
            "gpr": {
                "z": 2.0,
                "Menv": 100,
                "Ntrain": 40,
                "trainfrac": 1.0,
                "trainsel": "random",
            },
        },
        # andreagrisafi/SALTED-datasets: water_monomer_PySCF_subset100/README.md
        # 2026-07 Ntrain=40/validation=60 on 100-structure subset: % RMSE 8.216e-01
        validation_rmse_threshold=1.5,
    ),
]

MODELS: dict[str, ModelSpec] = {spec.key: spec for spec in _SPECS}


def params(specs) -> list:
    """Build a pytest parameter list, each param carrying its model's marker.

    Used with ``indirect=True`` so the ``model_spec`` fixture receives the key
    and every downstream fixture is cached per model by pytest itself.
    """
    return [pytest.param(spec.key, marks=getattr(pytest.mark, spec.marker)) for spec in specs]


""" Model groups: """

ALL_MODELS = params(MODELS.values())
"""Every model: the full training pipeline and the prediction step."""

LIVE_API_MODELS = params(s for s in MODELS.values() if not s.is_response)
"""Models covered by the in-process prediction API.

``salted.init_pred.build`` calls ``get_feats_projs``, not
``get_feats_projs_response``, and ``salted.salted_prediction.build`` has no
response branch -- so that API is density-only. Drop the filter once it grows
a density-response path.
"""

MPI_MODELS = params(s for s in MODELS.values() if s.check_mpi)
"""Models run twice, serially and under mpirun, for equivalence checking."""

DERIVED_PROPERTY_MODELS = params(s for s in MODELS.values() if s.has_derived_properties)
"""Models whose validation step also derives electrostatic observables
(total charge, dipole moment, Hartree energy) from the density coefficients."""

LIVE_API_MPI_MODELS = params(s for s in MODELS.values() if s.check_mpi and not s.is_response)
"""MPI-checked models the live prediction API also supports.

Used by the atom-parallel ``salted_prediction`` test, which exercises SALTED's
atom partitioning rather than anything dataset-specific -- so one model is
enough, and it must be one the density-only API can load.
"""
