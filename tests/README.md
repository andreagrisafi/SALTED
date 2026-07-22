# SALTED test suite

## Tests and Markers

- **Unit tests** in `tests/unit/`: fast, self-contained tests (no external data needed). Unmarked — select them by path (`pytest tests/unit`).
- **Example tests** in `tests/integration/`: end-to-end runs of the example ML pipelines plus prediction tests on the trained model, driven by precomputed data from the [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets) repository. All carry the `example` marker; see [Test scopes](#test-scopes) below.

  The four sub-markers select **disjoint** test sets (`example` is their union):

  | Marker | Dataset | Description |
  |---|---|---|
  | `aims` | `water_monomer_aims` | Serial pipeline + prediction tests |
  | `cp2k` | `water_monomer_CP2K_subset100` | Serial pipeline + prediction tests |
  | `pyscf` | `water_monomer_PySCF_subset100` | Serial pipeline + prediction tests |
  | `mpi` | `water_monomer_aims` | Serial-vs-MPI equivalence tests (pipeline + predictions) |

  - Reference data: 100-structures dataset, Ntrain=40, tested on 2026-07

  | example | nconf | Ntrain | % RMSE | threshold | time (1 core) |
  |---|---|---|---|---|---|
  | `water_monomer_aims` | 100 | 40 | 9.680e-01 | 1.5 | ~ 4 min |
  | `water_monomer_pyscf` | 100 | 40 | 7.990e-01 | 1.5 | ~ 4 min |
  | `water_monomer_cp2k` | 100 | 40 | 1.497e+00 | 2.5 | ~ 6 min |
  | `water_monomer_aims` (MPI) | 100 | 40 | 9.680e-01 | 1.5 | ~ 6 min |

> For detailed information on the tests, see the bottom of this README:
> [Test scopes](#test-scopes) and
> [Fixture design](#fixture-design).

## Running the Tests Locally

### Setup

The unit tests only need an editable install, then you can run from the project root.
The example tests need precomputed data from [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets) repository, or they will be skipped automatically with a warning.

```bash
# in the cloned SALTED directory:
pip install -e ".[test]"  # from the SALTED repository root
git clone https://github.com/andreagrisafi/SALTED-datasets.git ../SALTED-datasets  # put it aside the SALTED dir
```

Or, point `pytest` elsewhere with `--datasets-path /path/to/SALTED-datasets` or the `SALTED_DATASETS_PATH` environment variable.

### Running Tests

```bash
# only unit tests (fastest)
pytest tests/unit              # fast unit tests
# with SALTED-datasets
pytest -m example              # all example pipelines
pytest -m aims                 # only the aims serial tests
pytest -m pyscf                # only the pyscf serial tests
pytest -m cp2k                 # only the cp2k serial tests
pytest -m mpi                  # only the serial-vs-MPI equivalence aims tests
```

Options (see `pytest --help`, section "custom options"):

| Option | Default | Meaning |
|---|---|---|
| `--datasets-path PATH` | `$SALTED_DATASETS_PATH` or `../SALTED-datasets` | SALTED-datasets checkout |
| `--ntrain N` | each example's calibrated value (40) | reduce the training-set size for faster runs; a reduced run checks a loose 20 % RMSE bound instead of the calibrated thresholds |
| `--mpi-np N` | 2 | MPI tasks for the MPI equivalence test |
| `--require-datasets` | off | fail (instead of skip) when SALTED-datasets or `mpirun` are unavailable; used in CI so a missing dataset cannot silently pass |


The `pyscf` unit tests are skipped unless `pyscf` is installed and importable.

### Output Files

Each example pipeline runs in a fresh temporary workspace (dataset files are symlinked/copied in, `inp.yaml` is generated, all outputs are written there).
By default this lives under pytest's temp root:

```
/tmp/pytest-of-<user>/pytest-<N>/salted_<example>_0/
```

pytest keeps the last 3 runs there for inspection. To run in a **designated directory** instead (e.g. more disk space, a faster scratch mount, or a predictable path to inspect outputs) use pytest's built-in `--basetemp`:

```bash
pytest -m aims --basetemp /scratch/salted-tests
# -> /scratch/salted-tests/salted_water_monomer_aims_0/...
```


## Running the Tests in GitHub CI

The GitHub workflow [`.github/workflows/ci.yaml`](../.github/workflows/ci.yaml) runs this test suite:

| Event | `unit-tests` | `example-tests` (aims, pyscf, cp2k, mpi) |
|---|---|---|
| push to `master` or `dev_test_CI` | yes | yes, all four in parallel |
| push to any other branch | yes | skipped |
| PR into `master` or `dev_test_CI` | yes | yes, all four in parallel |
| manual dispatch (Actions tab, "Run workflow") | yes | yes, all four in parallel |

Notes:

- For SALTED-datasets, CI fetches only the required dataset directories and caches them by the upstream HEAD commit.
- CI always passes `--require-datasets`, so a missing dataset directory or `mpirun` fails the job instead of silently skipping, which is different from local behavior defaults.
- Each `example-tests` job has a 10-minute timeout (covering dependency install, dataset fetch, and the pipeline itself); if a job times out flakily, raise `timeout-minutes` in the workflow.

## Test scopes

### Serial tests (`aims`, `pyscf`, `cp2k`)

Each per-example marker runs the **same four tests** on its dataset:

1. **`test_example_pipeline`**
    - the eight pipeline steps as subprocesses (`python -m salted.<step>`, exactly as a user would), asserting each step's artifacts and the calibrated % RMSE threshold above.
2. **`test_prediction_matches_validation`**
    - the live-prediction API (`init_pred` + `salted.salted_prediction`, in-process): predicting every validation structure must reproduce `salted.validation`'s COEFFS files.
3. **`test_prediction_gradient_finite_difference`**
    -  same API with `gradient=True`: central finite differences must converge to the analytical gradient at second order.
4. **`test_prediction_step_matches_validation`**
    - the prediction pipeline step (`python -m salted.prediction`) run on the validation structures (sliced from the dataset xyz; `inp.yaml` temporarily retargeted with a separate `predname`, then restored): its coefficients must reproduce the validation step's.

Tests 2-4 reuse the trained model from test 1's workspace so that nothing is retrained.

### MPI tests (`mpi`)

SALTED's MPI tests verifies the following:

- **Pipeline equivalence**
    - rerun `sparse_descriptor`, `rkhs_vector`, `hessian_matrix`, `validation` under `mpirun -n 2`; regression matrices, weights, and RMSE must match the serial run with small differences.
- **`test_prediction_step_mpi_matches_serial`**
    - `salted.prediction` splits the prediction *structures* across ranks; an MPI prediction on validation set must match the cached serial run.
- **`test_salted_prediction_mpi_matches_serial`**
    - `salted.salted_prediction` splits the *atoms* of one structure across ranks; run via `_mpi_predict_driver.py`.

## Fixture design

Training a model costs 30 s - 2 min per backend, so the integration tests share trained workspaces through pytest fixtures (`tests/conftest.py`; injected by argument name, never imported). The dependency tree:

```
datasets_path (session)          locate SALTED-datasets, else skip (or fail in CI)
  └─ make_workspace (session)    factory: tmp workspace per example
       │                         (QM data symlinked read-only, xyz copied, inp.yaml generated)
       └─ serial_run (session, cached per example)
            │                    the full 8-step pipeline — runs ONCE per example
            ├─ test_example_pipeline                    artifacts + RMSE
            ├─ load_predictor (per-test)                trained model, in-process
            │    ├─ test_prediction_matches_validation
            │    └─ test_prediction_gradient_finite_difference
            ├─ valset_prediction (cached per example)   salted.prediction on the
            │    │                                      validation set — runs ONCE
            │    ├─ test_prediction_step_matches_validation
            │    └─ test_prediction_step_mpi_matches_serial   (mpi)
            ├─ test_salted_prediction_mpi_matches_serial      (mpi)
            └─ parallel_run (module)                    MPI rerun of the pipeline
                 ├─ test_regression_matrices_match            (mpi)
                 ├─ test_weights_match                        (mpi)
                 └─ test_validation_rmse_matches              (mpi)
```
