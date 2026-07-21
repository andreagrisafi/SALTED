# SALTED test suite

## Tests and Markers

- **Unit tests** in `tests/unit/`: fast, self-contained tests (no external data needed).
- **Example tests** in `tests/integration/`:
  - End-to-end runs of the example ML pipelines, `initialize` -> [intermediate steps] -> `validation`, plus `salted_prediction` on the trained model for values and gradients.
  - Precomputed data from the [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets) repository.
  - Each example additionally tests the live-prediction API (`salted.salted_prediction`) on the trained model: predictions must reproduce the validation-step coefficients, and central finite differences must converge to the analytical coefficient gradients at second order.

  | Marker | Dataset | Description |
  |---|---|---|
  | `aims` | `water_monomer_aims` | Serial pipeline + prediction tests + MPI equivalence tests |
  | `cp2k` | `water_monomer_CP2K_subset100` | Serial pipeline + prediction tests |
  | `pyscf` | `water_monomer_PySCF_subset100` | Serial pipeline + prediction tests |
  | `mpi` | `water_monomer_aims` | Rerun under `mpirun -n 2`, verify matrices/weights/RMSE match serial |

  - Reference data: 100-structures dataset, Ntrain=40, tested on 2026-07

  | example | nconf | Ntrain | % RMSE | threshold | time (1 core) |
  |---|---|---|---|---|---|
  | `water_monomer_aims` | 100 | 40 | 9.680e-01 | 1.5 | ~ 4 min |
  | `water_monomer_pyscf` | 100 | 40 | 7.990e-01 | 1.5 | ~ 4 min |
  | `water_monomer_cp2k` | 100 | 40 | 1.497e+00 | 2.5 | ~ 6 min |
  | `water_monomer_aims` (MPI) | 100 | 40 | 9.680e-01 | 1.5 | ~ 6 min |

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
# Examples:
pytest tests/unit              # fast unit tests
pytest -m example              # all example pipelines (needs SALTED-datasets)
pytest -m aims                 # only the aims example + aims MPI tests
pytest -m "aims and not mpi"   # only the aims serial pipeline
pytest -m cp2k                 # only the cp2k subset100 pipeline
pytest -m "example and not mpi"   # skip the MPI equivalence test
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

| Event | `unit-tests` | `example-tests` (aims, pyscf, cp2k) |
|---|---|---|
| push to `master` or `dev_test_CI` | yes | yes, all three in parallel |
| push to any other branch | yes | skipped |
| PR into `master` or `dev_test_CI` | yes | yes, all three in parallel |
| manual dispatch (Actions tab, "Run workflow") | yes | yes, all three in parallel |

Notes:

- For SALTED-datasets, CI fetches only the required dataset directories and caches them by the upstream HEAD commit.
- CI always passes `--require-datasets`, so a missing dataset directory or `mpirun` fails the job instead of silently skipping, which is different from local behavior defaults.
- Each `example-tests` job has a 10-minute timeout (covering dependency install, dataset fetch, and the pipeline itself); if a job times out flakily, raise `timeout-minutes` in the workflow.
