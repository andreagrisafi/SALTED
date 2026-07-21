"""Shared pytest fixtures and helpers for the SALTED test suite.

Two kinds of tests live under ``tests/``:

- ``tests/unit``: fast, self-contained tests of individual functions.
- ``tests/integration``: end-to-end runs of the example ML pipelines, using
  precomputed QM data (density-fitting coefficients and overlaps) from the
  SALTED-datasets repository (https://github.com/andreagrisafi/SALTED-datasets).

Integration tests are marked ``example`` and are skipped automatically when the
datasets repository is not available (with ``--require-datasets``, as used in
CI, they fail loudly instead). Point to it with::

    pytest --datasets-path /path/to/SALTED-datasets

or the ``SALTED_DATASETS_PATH`` environment variable (the default is a sibling
directory ``../SALTED-datasets`` next to the SALTED repository).

The training-set size of the integration runs can be reduced for quick local
runs with ``--ntrain N`` (default: each example's calibrated value).
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


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

EXAMPLES = {
    "water_monomer_aims": {
        "dataset_dir": "water_monomer_aims",
        "nconf": 100,
        "inp": {
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
        # angreagrisafi/SALTED-datasets: water_monomer_aims/README.md
        # validation RMSE 9.680e-01 %
        "rmse_threshold": 1.5,
    },
    "water_monomer_cp2k": {
        "dataset_dir": "water_monomer_CP2K_subset100",
        "nconf": 100,
        # CP2K ships no basis_data.yaml; it ships the RI basis files with ({species}-{dfbasis}, e.g. H-RI-basis)
        "parse_basis": True,
        "inp": {
            "salted": {"saltedname": "test", "saltedpath": "./", "verbose": True},
            "system": {"filename": "./water_monomers_100.xyz", "species": ["H", "O"]},
            "qm": {
                "path2qm": "./",
                "qmcode": "cp2k",
                "periodic": "3D",
                "dfbasis": "RI-basis",
                "coeffile": "dummy-RI_DENSITY_COEFFS.dat",
                "ovlpfile": "dummy-RI_2C_INTS.fm",
                "pseudocharge": [1.0, 6.0],
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
                "trainsel": "sequential",
            },
        },
        # angreagrisafi/SALTED-datasets: water_monomer_CP2K_subset100/README.md
        # 2026-07 Ntrain=40/validation=60 on 100-structure subset: % RMSE 1.497e+00
        "rmse_threshold": 2.5,
    },
    "water_monomer_pyscf": {
        "dataset_dir": "water_monomer_PySCF_subset100",
        "nconf": 100,
        "inp": {
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
                "trainsel": "sequential",
            },
        },
        # angreagrisafi/SALTED-datasets: water_monomer_PySCF_subset100/README.md
        # 2026-07 Ntrain=40/validation=60 on 100-structure subset: % RMSE 7.990e-01
        "rmse_threshold": 1.5,
    },
}


def pytest_addoption(parser):
    parser.addoption(
        "--datasets-path",
        default=os.environ.get("SALTED_DATASETS_PATH", str(REPO_ROOT.parent / "SALTED-datasets")),
        help="Path to a checkout of the SALTED-datasets repository "
        "(default: $SALTED_DATASETS_PATH or ../SALTED-datasets)",
    )
    parser.addoption(
        "--ntrain",
        type=int,
        default=None,
        help="Override gpr.Ntrain of the example pipelines (smaller = faster). "
        "Default: the full value from each example's inp.yaml.",
    )
    parser.addoption(
        "--mpi-np",
        type=int,
        default=2,
        help="Number of MPI tasks for the MPI equivalence test (default: 2)",
    )
    parser.addoption(
        "--require-datasets",
        action="store_true",
        default=False,
        help="Fail (instead of skip) integration tests when SALTED-datasets or "
        "mpirun are unavailable. Meant for CI, where a missing dataset must be "
        "a loud error, not a silently green job.",
    )


def skip_or_fail(require_datasets: bool, msg: str):
    """Skip with proper message, or fail loudly when --require-datasets is set."""
    if require_datasets:
        pytest.fail(f"--require-datasets: {msg}")
    pytest.skip(msg)


# markers are declared once in pyproject.toml [tool.pytest.ini_options]

@pytest.fixture(scope="session")
def datasets_path(request) -> Path:
    """Root of the SALTED-datasets checkout; skips (or fails) dependent tests if absent."""
    path = Path(request.config.getoption("--datasets-path")).resolve()
    if not path.is_dir():
        skip_or_fail(
            request.config.getoption("--require-datasets"),
            f"SALTED-datasets not found at {path} (use --datasets-path or $SALTED_DATASETS_PATH)",
        )
    return path


@pytest.fixture(scope="session")
def mpirun_cmd(request):
    """Base mpirun command; skips (or fails with --require-datasets) if no MPI launcher."""
    import shutil

    mpirun = shutil.which("mpirun")
    if mpirun is None:
        skip_or_fail(request.config.getoption("--require-datasets"), "mpirun not available")
    cmd = [mpirun]
    version = subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout
    if "Open MPI" in version or "OpenRTE" in version:
        # allow more ranks than cores on small CI runners
        cmd.append("--oversubscribe")
    return cmd


class PipelineWorkspace:
    """A temp working directory wired up to run one example's ML pipeline."""

    def __init__(self, name: str, root: Path, datasets_path: Path, ntrain: int | None, require_datasets: bool = False):
        self.name = name
        self.root = root
        spec = EXAMPLES[name]
        self.dataset = datasets_path / spec["dataset_dir"]
        if not self.dataset.is_dir():
            skip_or_fail(require_datasets, f"dataset {self.dataset} not found")

        # QM data: symlink (read-only use), xyz files: copy
        for sub in ("coefficients", "overlaps"):
            src = self.dataset / sub
            assert src.is_dir(), f"missing {src}"
            (root / sub).symlink_to(src)
        for xyz in self.dataset.glob("*.xyz"):
            (root / xyz.name).write_bytes(xyz.read_bytes())

        # inp.yaml: deep copy of the template, + external basis + optional Ntrain
        self.inp = yaml.safe_load(yaml.safe_dump(spec["inp"]))  # deep copy

        # Density-fitting basis: SALTED reads/writes it via inp.qm.dfbasis_file or parse basis files
        self.basis_prelude: list[str] = []
        if spec.get("parse_basis"):
            # CP2K: move RI basis files into the workspace, then SALTED can parse them.
            # only supports CP2K, add other codes if needed
            assert self.inp["qm"]["qmcode"] == "cp2k", (
                f"parse_basis is only implemented for qmcode 'cp2k' "
                f"(example {name} has qmcode {self.inp['qm']['qmcode']!r})"
            )
            dfbasis = self.inp["qm"]["dfbasis"]
            for spe in self.inp["system"]["species"]:
                src = self.dataset / f"{spe}-{dfbasis}"
                if not src.is_file():
                    skip_or_fail(
                        require_datasets,
                        f"dataset {self.dataset.name} does not ship RI basis file "
                        f"{src.name}; needed to derive the density-fitting basis",
                    )
                (root / src.name).write_bytes(src.read_bytes())
            self.basis_fpath = root / "basis_data.yaml"
            self.basis_fpath.write_text("")  # empty registry, populated by get_basis_info
            self.basis_prelude = ["get_basis_info"]
        else:
            # aims / PySCF: load from the dataset's prepared basis_data.yaml
            self.basis_fpath = self.dataset / "basis_data.yaml"
            if not self.basis_fpath.is_file():
                skip_or_fail(
                    require_datasets,
                    f"dataset {self.dataset.name} does not ship basis_data.yaml "
                    f"(expected at {self.basis_fpath}); it must carry the lmax/nmax "
                    "info of its density-fitting basis",
                )
        self.inp["qm"]["dfbasis_file"] = str(self.basis_fpath)

        self.ntrain_reduced = ntrain is not None and ntrain != self.inp["gpr"]["Ntrain"]
        if ntrain is not None:
            self.inp["gpr"]["Ntrain"] = ntrain
        inp_text = yaml.safe_dump(self.inp, sort_keys=False)
        (root / "inp.yaml").write_text(inp_text)
        print(f"\n=== {name}: inp.yaml (workspace {root}) ===\n{inp_text}=== end inp.yaml ===")

        self.timings: dict[str, float] = {}
        self.rmse: float | None = None

    @property
    def ntrain(self) -> int:
        return int(self.inp["gpr"]["Ntrain"] * self.inp["gpr"]["trainfrac"])

    @property
    def saltedname(self) -> str:
        return self.inp["salted"]["saltedname"]

    @property
    def mz(self) -> str:
        """The M{Menv}_zeta{z} path component used by several steps."""
        return f"M{self.inp['gpr']['Menv']}_zeta{self.inp['gpr']['z']}"

    @property
    def reg_str(self) -> str:
        """The reg{log10(regul)} path component (e.g. 'reg-6'); regul defaults to 1e-6."""
        regul = self.inp["gpr"].get("regul", 1e-6)
        return f"reg{int(np.log10(regul))}"

    @property
    def nconf(self) -> int:
        return EXAMPLES[self.name]["nconf"]

    @property
    def rmse_threshold(self) -> float:
        if self.ntrain_reduced:
            # loose sanity bound (%) for --ntrain reduced runs
            return 20.0
        return EXAMPLES[self.name]["rmse_threshold"]

    def run_step(self, module: str, mpi: list[str] | None = None, timeout: float = 7200):
        """Run ``python -m salted.<module>`` in the workspace and time it.

        Runs with ``HDF5_USE_FILE_LOCKING=FALSE``: HDF5 guards file access
        with an advisory ``flock(2)``, which fails spuriously (EAGAIN /
        ``BlockingIOError``, or ENOLCK "No locks available") on filesystems
        with broken or absent lock support (NFS, some parallel/CI
        filesystems), even with a single writer. SALTED already serialises
        HDF5 access via MPI barriers and rank-0 guards, and the HDF5 docs
        state the lock can be safely disabled when writes are not concurrent:
        https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html
        """
        cmd = [sys.executable, "-m", f"salted.{module}"]
        if mpi:
            cmd = mpi + cmd
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)
        elapsed = time.perf_counter() - t0
        label = module if not mpi else f"{module}[mpi]"
        self.timings[label] = elapsed
        assert proc.returncode == 0, (
            f"salted.{module} failed (exit {proc.returncode}) after {elapsed:.1f}s\n"
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )
        return proc.stdout

    def parse_rmse(self, stdout: str) -> float:
        """Extract the final '% RMSE: x.xxxe+xx' printed by salted.validation."""
        matches = re.findall(r"%\s*RMSE:\s*([0-9.eE+-]+)", stdout)
        assert matches, f"no '% RMSE' line found in validation output:\n{stdout[-2000:]}"
        self.rmse = float(matches[-1])
        return self.rmse

    def timing_report(self) -> str:
        lines = [f"--- {self.name} pipeline timings (Ntrain={self.ntrain}) ---"]
        lines += [f"{k:>24s}: {v:8.1f} s" for k, v in self.timings.items()]
        lines.append(f"{'total':>24s}: {sum(self.timings.values()):8.1f} s")
        if self.rmse is not None:
            lines.append(f"{'% RMSE':>24s}: {self.rmse:.3e}")
        return "\n".join(lines)


@pytest.fixture(scope="session")
def make_workspace(request, datasets_path, tmp_path_factory):
    """Factory building a ready-to-run pipeline workspace for an example."""

    def _make(example: str) -> PipelineWorkspace:
        root = tmp_path_factory.mktemp(f"salted_{example}_")
        return PipelineWorkspace(
            example,
            root,
            datasets_path,
            request.config.getoption("--ntrain"),
            require_datasets=request.config.getoption("--require-datasets"),
        )

    return _make


SERIAL_PIPELINE = [
    "initialize",
    "sparse_selection",
    "sparse_descriptor",
    "rkhs_projector",
    "rkhs_vector",
    "hessian_matrix",
    "solve_regression",
    "validation",
]


def run_full_pipeline(ws: PipelineWorkspace, mpi: list[str] | None = None, mpi_steps: tuple = ()) -> PipelineWorkspace:
    """Run all pipeline steps; steps named in ``mpi_steps`` run under mpirun.

    Any basis-prelude steps (e.g. ``get_basis_info`` for CP2K, which derives the
    density-fitting basis from the shipped RI basis files) run first, serially.
    """
    for step in ws.basis_prelude + SERIAL_PIPELINE:
        stdout = ws.run_step(step, mpi=mpi if step in mpi_steps else None)
        if step == "validation":
            ws.parse_rmse(stdout)
    return ws


@pytest.fixture(scope="session")
def pipeline_runner():
    """Expose run_full_pipeline to test modules (avoids importing conftest)."""
    return run_full_pipeline


@pytest.fixture(scope="session")
def serial_run(make_workspace):
    """Run each example's full pipeline serially, once per session, on demand."""
    cache: dict[str, PipelineWorkspace] = {}

    def _get(example: str) -> PipelineWorkspace:
        if example not in cache:
            ws = make_workspace(example)
            run_full_pipeline(ws)  # raises on failure -> only successes cached
            print("\n" + ws.timing_report())
            cache[example] = ws
        return cache[example]

    return _get
