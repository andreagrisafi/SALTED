"""Shared pytest fixtures for the SALTED test suite. See tests/README.md.

Integration tests need a SALTED-datasets checkout and skip without one, or fail under
``--require-datasets`` (CI, where a missing dataset must not pass quietly).
"""

import os
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml
from models import MODELS, ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]

MPI_STEPS = ("sparse_descriptor", "rkhs_vector", "hessian_matrix", "validation")


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
        help="Override gpr.Ntrain of the model pipelines (smaller = faster). "
        "Default: the full value from each model's spec.",
    )
    parser.addoption(
        "--mpi-np",
        type=int,
        default=2,
        help="Number of MPI tasks for the MPI equivalence tests (default: 2)",
    )
    parser.addoption(
        "--require-datasets",
        action="store_true",
        default=False,
        help="Fail (instead of skip) integration tests when SALTED-datasets is "
        "unavailable. Meant for CI, where a missing dataset must be a loud "
        "error, not a silently green job.",
    )


def skip_or_fail(require_datasets: bool, msg: str):
    """Skip with proper message, or fail loudly when --require-datasets is set."""
    if require_datasets:
        pytest.fail(f"--require-datasets: {msg}")
    pytest.skip(msg)


# markers are declared once in pyproject.toml [tool.pytest.ini_options]


def pytest_configure(config):
    """Fail at startup if a model's marker was never declared in pyproject.toml"""
    declared = {entry.split(":")[0].strip() for entry in config.getini("markers")}
    missing = sorted({spec.marker for spec in MODELS.values()} - declared)
    if missing:
        raise pytest.UsageError(
            f"model markers not declared in pyproject.toml [tool.pytest.ini_options] "
            f"markers: {missing}"
        )


class PipelineWorkspace:
    """A temp working directory wired up to run one model's ML pipeline.

    Owns the model spec, the staged inputs, the SALTED output layout, and the
    results of running it (timings, metrics).
    """

    def __init__(
        self,
        spec: ModelSpec,
        root: Path,
        datasets_path: Path,
        ntrain: int | None,
        require_datasets: bool = False,
    ):
        self.spec = spec
        self.root = root
        self.dataset = datasets_path / spec.dataset_dir
        if not self.dataset.is_dir():
            skip_or_fail(require_datasets, f"dataset {self.dataset} not found")

        self._stage_qm_data(require_datasets)
        self._write_inp(ntrain, require_datasets)

        self.timings: dict[str, float] = {}
        self.validation_rmse: float | None = None

    def _stage_qm_data(self, require_datasets: bool):
        """Link the dataset's QM data into the workspace, copy its xyz files.
        """
        for sub in ("coefficients", "overlaps"):
            src = self.dataset / sub
            assert src.is_dir(), f"missing {src}"
            dst = self.root / sub
            dst.mkdir()
            for f in src.iterdir():
                if f.name == "averages":  # exclude
                    continue
                (dst / f.name).symlink_to(f)  # Entries are linked whether they are files or directories

        for xyz in self.dataset.glob("*.xyz"):
            (self.root / xyz.name).write_bytes(xyz.read_bytes())

        if self.spec.inp["qm"]["qmcode"] == "cp2k":
            src = self.dataset / "basis"
            if not src.is_dir():
                skip_or_fail(
                    require_datasets,
                    f"dataset {self.dataset.name} does not ship a basis/ directory "
                    "with the alphas/contra .dat files of its density-fitting basis; "
                    "needed for the charge/dipole moment integrals of the validation step",
                )
            # Real dir of per-file symlinks
            dst = self.root / "basis"
            dst.mkdir()
            for f in src.iterdir():
                (dst / f.name).symlink_to(f)
            pseudos = list(self.dataset.glob("*-local_pseudo.dat"))
            assert pseudos, (
                f"dataset {self.dataset.name} does not ship {{spe}}-local_pseudo.dat "
                "files; salted.validation reads them for the charge/dipole/Hartree checks"
            )
            for f in pseudos:
                (dst / f.name).symlink_to(f)

    def _write_inp(self, ntrain: int | None, require_datasets: bool):
        """Build inp.yaml from the spec: deep copy + external basis + Ntrain."""
        self.inp = yaml.safe_load(yaml.safe_dump(self.spec.inp))  # deep copy

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
        (self.root / "inp.yaml").write_text(inp_text)
        print(f"\n=== {self.name}: inp.yaml (workspace {self.root}) ===\n{inp_text}=== end inp.yaml ===")

    @property
    def name(self) -> str:
        return self.spec.key

    @property
    def nconf(self) -> int:
        return self.spec.nconf

    @property
    def saltedtype(self) -> str:
        return self.spec.saltedtype

    @property
    def is_response(self) -> bool:
        return self.spec.is_response

    @property
    def cart_components(self) -> tuple[str, ...]:
        return self.spec.cart_components

    @property
    def weights_rtol(self) -> float:
        return self.spec.weights_rtol

    @property
    def validation_rmse_threshold(self) -> float:
        if self.ntrain_reduced:
            # loose sanity bound (%) for --ntrain reduced runs
            return 20.0
        return self.spec.validation_rmse_threshold

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
    def equirepr_dir(self) -> Path:
        """Where the descriptors, sparse selection and projectors are written."""
        return self.root / f"equirepr_{self.saltedname}"

    @property
    def rkhs_vector_dir(self) -> Path:
        """Where salted.rkhs_vector writes its psi-nm_conf*.npz files."""
        return self.root / f"rkhs-vectors_{self.saltedname}" / self.mz

    @property
    def regression_dir(self) -> Path:
        """Where the regression matrices and GPR weights are written."""
        return self.root / f"regrdir_{self.saltedname}" / self.mz

    def regression_path(self, name: str) -> Path:
        """Path of a regression artifact: 'Avec', 'Bmat' or 'weights'."""
        if name == "weights":
            return self.regression_dir / f"weights_N{self.ntrain}_{self.reg_str}.npy"
        return self.regression_dir / f"{name}_N{self.ntrain}.npy"

    def regression_array(self, name: str) -> np.ndarray:
        """Load a regression artifact: 'Avec', 'Bmat' or 'weights'."""
        return np.load(self.regression_path(name))

    @property
    def validation_output_dir(self) -> Path:
        """Where salted.validation writes its COEFFS-<iconf+1>.dat files
        (1-based index into the full dataset)."""
        return self.root / f"validations_{self.saltedname}" / self.mz / f"N{self.ntrain}_{self.reg_str}"

    def validation_property(self, name: str) -> np.ndarray:
        """Load validation output, for 'errors', also 'charges', 'dipoles, 'electrostatic_energy' for derived ones."""
        return np.loadtxt(self.validation_output_dir / f"{name}.dat", ndmin=2)

    def prediction_output_dir(self, predname: str) -> Path:
        """Where salted.prediction writes its COEFFS-<k+1>.dat files
        (1-based position in the prediction xyz)."""
        return (
            self.root / f"predictions_{self.saltedname}_{predname}" / self.mz / f"N{self.ntrain}_{self.reg_str}"
        )

    def coeff_dirs(self, base: Path) -> list[Path]:
        """COEFFS directories under ``base``: ``base`` itself for ``density``,
        one per Cartesian component of dn/dE for ``density-response``."""
        if self.is_response:
            return [base / icart for icart in self.cart_components]
        return [base]

    def coeff_files(self, base: Path, index: int) -> list[Path]:
        """COEFFS paths for structure ``index`` (1-based) under ``base``.
        """
        return [d / f"COEFFS-{index}.dat" for d in self.coeff_dirs(base)]

    def validation_indices(self) -> np.ndarray:
        """Sorted 0-based indices of the validation structures."""
        train_fpath = self.root / f"regrdir_{self.saltedname}" / f"training_set_N{self.inp['gpr']['Ntrain']}.txt"
        trainrangetot = np.loadtxt(train_fpath, dtype=int)
        return np.setdiff1d(np.arange(self.nconf), trainrangetot)

    def run_python(self, args: list[str], label: str, mpi: list[str] | None = None, timeout: float = 7200):
        """Run ``python <args...>`` in the workspace and time it. Returns stdout.

        Args:
            args: arguments after the interpreter, e.g. ``["-m", "salted.validation"]``.
            label: key under which the elapsed time is recorded in ``self.timings``,
                suffixed ``[mpi]`` when launched under mpirun.
            mpi: launcher prefix to prepend, e.g. ``["mpirun", "-n", "2"]``.
            timeout: seconds before the subprocess is killed.

        Notes:
            Runs with ``HDF5_USE_FILE_LOCKING=FALSE``: HDF5 guards file access
            with an advisory ``flock(2)``, which fails spuriously (EAGAIN /
            ``BlockingIOError``, or ENOLCK "No locks available") on filesystems
            with broken or absent lock support (NFS, some parallel/CI
            filesystems), even with a single writer. SALTED already serialises
            HDF5 access via MPI barriers and rank-0 guards, and the HDF5 docs
            state the lock can be safely disabled when writes are not concurrent:
            https://support.hdfgroup.org/documentation/hdf5/latest/_file_lock.html
        """
        cmd = [sys.executable] + args
        if mpi:
            cmd = mpi + cmd
            label = f"{label}[mpi]"
        env = os.environ.copy()
        if mpi:
            #avoid intelmpi / openmpi PATHs conflict
            mpi_dir = str(Path(mpi[0]).resolve().parent)
            env["PATH"] = mpi_dir + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=self.root, capture_output=True, text=True, timeout=timeout, env=env)
        elapsed = time.perf_counter() - t0
        self.timings[label] = elapsed
        assert proc.returncode == 0, (
            f"{label} failed (exit {proc.returncode}) after {elapsed:.1f}s\n"
            f"--- stdout (tail) ---\n{proc.stdout[-3000:]}\n"
            f"--- stderr (tail) ---\n{proc.stderr[-3000:]}"
        )
        return proc.stdout

    def run_step(self, module: str, mpi: list[str] | None = None, timeout: float = 7200):
        """Run ``python -m salted.<module>`` in the workspace and time it.

        Args:
            module: submodule of ``salted`` to run, e.g. ``"validation"``. Also
                the timing label.
            mpi: launcher prefix, or None to run serially. See ``run_python``.
            timeout: seconds before the subprocess is killed.
        """
        return self.run_python(["-m", f"salted.{module}"], label=module, mpi=mpi, timeout=timeout)

    def run_pipeline(self, mpi: list[str] | None = None, mpi_steps: tuple[str, ...] = ()):
        """Run the model's pipeline steps in order, taken from the spec.

        Args:
            mpi: launcher prefix, e.g. ``["mpirun", "-n", "2"]``. None runs
                everything serially, whatever ``mpi_steps`` says.
            mpi_steps: which steps to launch with it. Only some steps have an
                MPI code path, so the whole pipeline cannot go under mpirun;
                pass the ``MPI_STEPS`` constant.
        """
        assert set(mpi_steps) <= set(self.spec.steps), (
            f"{self.name}: mpi_steps {sorted(set(mpi_steps) - set(self.spec.steps))} "
            f"are not part of this model's pipeline"
        )
        for step in self.spec.steps:
            stdout = self.run_step(step, mpi=mpi if step in mpi_steps else None)
            if step == "validation":
                self.parse_validation_rmse(stdout)
        print("\n" + self.timing_report())
        return self

    def parse_validation_rmse(self, stdout: str) -> float:
        """Extract the final '% RMSE: x.xxxe+xx' printed by salted.validation."""
        matches = re.findall(r"%\s*RMSE:\s*([0-9.eE+-]+)", stdout)
        assert matches, f"no '% RMSE' line found in validation output:\n{stdout[-2000:]}"
        self.validation_rmse = float(matches[-1])
        return self.validation_rmse

    def timing_report(self) -> str:
        lines = [f"--- {self.name} pipeline timings (Ntrain={self.ntrain}) ---"]
        lines += [f"{k:>24s}: {v:8.1f} s" for k, v in self.timings.items()]
        lines.append(f"{'total':>24s}: {sum(self.timings.values()):8.1f} s")
        if self.validation_rmse is not None:
            lines.append(f"{'% RMSE':>24s}: {self.validation_rmse:.3e}")
        return "\n".join(lines)

    def write_prediction_from_validation(self) -> Path:
        """Write the validation structures as a prediction xyz; return its path."""
        from ase.io import read, write

        frames = read(self.root / self.inp["system"]["filename"], ":")
        fpath = self.root / "prediction_set_from_validation_set.xyz"
        write(fpath, [frames[i] for i in self.validation_indices()])
        return fpath

    @contextmanager
    def swap_prediction_inp(self, filename: str, predname: str):
        """Temporarily retarget inp.prediction; always restore inp.yaml.

        ParseConfig hard-codes ``<cwd>/inp.yaml``, so the modified copy must
        temporarily *be* inp.yaml. The workspace is session-shared, hence the
        guaranteed restore. ``self.inp`` is left untouched.
        """
        inp_fpath = self.root / "inp.yaml"
        original = inp_fpath.read_text()
        modified = yaml.safe_load(original)
        modified.setdefault("prediction", {}).update(filename=filename, predname=predname)
        try:
            inp_fpath.write_text(yaml.safe_dump(modified, sort_keys=False))
            yield
        finally:
            inp_fpath.write_text(original)


@dataclass(frozen=True)
class WorkspaceBuilder:
    """Builds a fresh workspace per call, with the session's CLI options baked in."""

    datasets_path: Path
    tmp_factory: pytest.TempPathFactory
    ntrain: int | None
    np_tasks: int
    require_datasets: bool
    mpirun: list[str]

    def build(self, spec: ModelSpec) -> PipelineWorkspace:
        root = self.tmp_factory.mktemp(f"salted_{spec.key}_")
        return PipelineWorkspace(spec, root, self.datasets_path, self.ntrain, self.require_datasets)

    @property
    def mpi_cmd(self) -> list[str]:
        return self.mpirun + ["-n", str(self.np_tasks)]


@pytest.fixture(scope="session")
def workspaces(request, tmp_path_factory) -> WorkspaceBuilder:
    """Session-wide workspace builder; skips integration tests without datasets."""
    require = request.config.getoption("--require-datasets")
    datasets_path = Path(request.config.getoption("--datasets-path")).resolve()
    if not datasets_path.is_dir():
        skip_or_fail(
            require,
            f"SALTED-datasets not found at {datasets_path} "
            "(use --datasets-path or $SALTED_DATASETS_PATH)",
        )

    # due to above change of env PATH, ensure again that mpirun is found in the new PATH
    mpirun = shutil.which("mpirun", path=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    assert mpirun is not None, (
        "mpirun not found on PATH; the integration tests require an MPI launcher"
    )
    cmd = [mpirun]
    version = subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout
    if "Open MPI" in version or "OpenRTE" in version:
        # allow more ranks than cores on small CI runners
        cmd.append("--oversubscribe")

    return WorkspaceBuilder(
        datasets_path=datasets_path,
        tmp_factory=tmp_path_factory,
        ntrain=request.config.getoption("--ntrain"),
        np_tasks=request.config.getoption("--mpi-np"),
        require_datasets=require,
        mpirun=cmd,
    )


@pytest.fixture(scope="session")
def model_spec(request) -> ModelSpec:
    """The model under test, supplied by ``indirect=True`` parametrization.
    """
    return MODELS[request.param]


@pytest.fixture(scope="session")
def serial_run(model_spec, workspaces) -> PipelineWorkspace:
    """Train the model serially, once per session. The reference for everything."""
    return workspaces.build(model_spec).run_pipeline()


@pytest.fixture(scope="session")
def mpi_run(model_spec, workspaces) -> PipelineWorkspace:
    """Train the same model again with the MPI-parallel steps under mpirun.

    A second workspace from the same spec, so any difference is attributable
    to the MPI code alone.
    """
    ws = workspaces.build(model_spec)
    return ws.run_pipeline(mpi=workspaces.mpi_cmd, mpi_steps=MPI_STEPS)
