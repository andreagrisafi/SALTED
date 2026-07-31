# SALTED test suite

## Design

Two tiers of tests:

- **Unit tests** (`tests/unit/`): fast, self-contained, no external data.
- **Integration tests** (`tests/integration/`): end-to-end runs of the ML pipeline on precomputed QM data from [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets). Skipped automatically when that repository is absent.

Concepts in the integration tests design:

- **Dataset**: from GitHub repo [andreagrisafi/SALTED-datasets](http://github.com/andreagrisafi/SALTED-datasets). Provides the QM data for a model pipeline.
- **Model**: defined by a dataset and an `inp.yaml` that trains it (see  `ModelSpec` in [`tests/models.py`](models.py)). Each model has a unique **marker**. Each model will be trained as a pipeline test, and then used for different subsequent tests.

| marker | dataset | `saltedtype`  | time (1 core) |
|---|---|---|---|---|---|
| `aims_density` | `water_monomer_AIMS` | `density` | ~4 min, ~10 with MPI |
| `pyscf_density` | `water_monomer_PySCF_subset100` | `density` | ~4 min |
| `cp2k_density` | `water_monomer_CP2K_subset100` | `density` | ~6 min |
| `aims_response` | `water_monomer_AIMS_response_subset100` | `density-response` | ~20 min with MPI |

The four markers select **disjoint** sets and `integration` is their union, so CI runs every test exactly once.

Two conventions worth knowing before reading the code:

- **`PipelineWorkspace` (`ws`) holds the state**: temp directory, artifact layout, per-step timings, RMSE. Ask `ws` for a path or an array (`ws.regression_array("Bmat")`) rather than rebuilding path strings, or for test results (`ws.validation_rmse`).
- **Fixtures are parametrized, not factories**: everything hangs off `model_spec` with `indirect=True`, so pytest caches each trained model per session and builds only what `-m` selects.

Rationale for individual tests and tolerances lives in their docstrings; `tests/conftest.py` and `tests/models.py` document the fixture and registry design.

## Running the tests

```bash
pip install -e ".[test]"  # in the SALTED directory
git clone https://github.com/andreagrisafi/SALTED-datasets.git ../SALTED-datasets
```

Or point elsewhere with `--datasets-path /path/to/SALTED-datasets` or set env var `$SALTED_DATASETS_PATH`.

```bash
pytest tests/unit                     # fast, no data needed
pytest -m integration                 # every model
pytest -m aims_density                # one model
pytest -m aims_density -k mpi         # just its serial-vs-MPI checks
pytest -m aims_density -k "not mpi"   # just its serial checks
```

| option | default | meaning |
|---|---|---|
| `--datasets-path PATH` | `$SALTED_DATASETS_PATH` or `../SALTED-datasets` | SALTED-datasets checkout |
| `--ntrain N` | 40 | smaller training set for faster runs; checks a loose 20 % RMSE bound instead of the calibrated threshold |
| `--mpi-np N` | 2 | MPI tasks for the equivalence tests |
| `--require-datasets` | off | fail instead of skip when SALTED-datasets is missing; used in CI |

Each model runs in a fresh temp workspace under `/tmp/pytest-of-<user>/pytest-<N>/salted_<model>_0/`. Use `--basetemp /some/dir` to put them somewhere predictable. The `pyscf` unit tests are skipped unless `pyscf` is installed.

Do not run a marker's tests concurrently (no `pytest-xdist`): the prediction tests write into the shared trained workspace.

## Coverage

Which `salted.*` modules the integration tests exercise. `+MPI` means the module is also rerun under `mpirun` and compared against the serial result.

| `salted.*` | aims_density | pyscf_density | cp2k_density | aims_response |
|---|:--:|:--:|:--:|:--:|
| `initialize` → `wigner`, `scalar_vector` | y | y | y | y +antisymm |
| `sparsify_features` | n/a | y | y | n/a |
| `sparse_selection` | y | y | y | y |
| `sparse_descriptor` | y +MPI | y | y | y +MPI |
| `rkhs_projector` | y | y | y | y response variant |
| `rkhs_vector` | y +MPI | y | y | y +MPI |
| `hessian_matrix` → `numba_sparse`, `get_averages` | y +MPI | y | y | y +MPI, no averages |
| `solve_regression` | y | y | y | y |
| `validation` | y +MPI | y | y | y +MPI |
| `prediction` | y +MPI | y | y | y +MPI |
| `init_pred` + `salted_prediction` | y +MPI | y | y | n (density-only API)

The `n/a` entries follow from the model configuration, not from implementation gaps.
`salted.minimize_loss` is deliberately not covered.

## Continuous integration

[`.github/workflows/ci.yaml`](../.github/workflows/ci.yaml), one job per model:

| event | `unit-tests` | `integration-tests` |
|---|---|---|
| push to `master` or `dev_test_CI` | yes | yes, all four in parallel |
| push to any other branch | yes | skipped |
| PR into `master` or `dev_test_CI` | yes | yes, all four in parallel |
| manual dispatch (Actions tab) | yes | yes, all four in parallel |

CI fetches only the dataset directories it needs and caches them by upstream HEAD. It always passes `--require-datasets`, so a missing dataset fails the job rather than silently skipping, unlike local runs.

## Adding a model

1. a `ModelSpec` in [`tests/models.py`](models.py): dataset directory, marker, `inp.yaml`, calibrated `validation_rmse_threshold`
2. its marker in `pyproject.toml` under `[tool.pytest.ini_options] markers`
3. its marker in the `ci.yaml` job matrix, and its dataset in that workflow's sparse-checkout

`ModelSpec.steps` defaults to the shared `TRAINING_PIPELINE`, so every model provably runs the same code path unless it says otherwise in one visible line. Both the runner (`ws.run_pipeline`) and the artifact checks (`STEP_CHECKS` in `test_pipeline.py`) read that list, so a model that skips a step also skips its artifact check, which is how an alternative like `minimize_loss` would slot in.

## Adding a test

Pick a model group from [`models.py`](models.py), parametrize on `model_spec` with `indirect=True`, and ask for `serial_run`, or for `mpi_run` too if you are comparing the two:

```python
from models import ALL_MODELS

@pytest.mark.parametrize("model_spec", ALL_MODELS, indirect=True)
def test_weights_are_finite(serial_run):
    assert np.all(np.isfinite(serial_run.regression_array("weights")))
```

The group carries each model's marker, so the test needs none of its own and reuses the model that marker's CI job already trained. Get paths and arrays from `ws` instead of building path strings.
