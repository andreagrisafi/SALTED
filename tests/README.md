# SALTED test suite

## Tests and Markers

- **Unit tests** in `tests/unit/`: fast, self-contained tests (no external data needed).
- **Example tests** in `tests/integration/`: end-to-end runs of the example ML pipelines (`initialize` -> intermediate steps -> `validation`).
  - Precomputed data from the [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets) repository. Will be skipped automatically when the datasets are absent.

  | Marker | Dataset | Description |
  |---|---|---|
  | `aims` | `water_monomer_aims` | Serial pipeline + MPI equivalence tests |
  | `cp2k` | `water_monomer_CP2K_subset100` | Serial pipeline (+ `get_basis_info` prelude) |
  | `pyscf` | `water_monomer_PySCF_subset100` | Serial pipeline |
  | `mpi` | `water_monomer_aims` | Rerun under `mpirun -n 2`, verify matrices/weights/RMSE match serial |

  Timing for 1 core, approximate. Of the 100 structures, 40 are used for training and 60 for validation.
  All integration tests are also marked `example`; MPI tests carry both `aims` and `mpi`.

## Setup

The unit tests need nothing beyond an editable install.
The example tests need precomputed data from [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets) repository.

```bash
# in the cloned SALTED directory:
pip install -e ".[test]"  # from the SALTED repository root
git clone https://github.com/andreagrisafi/SALTED-datasets.git ../SALTED-datasets  # put it aside the SALTED dir
```

Or, point `pytest` elsewhere with `--datasets-path /path/to/SALTED-datasets` or the `SALTED_DATASETS_PATH` environment variable.

## Running

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
| `--ntrain N` | full example value | reduce the training-set size for faster runs |
| `--mpi-np N` | 2 | MPI tasks for the MPI equivalence test |


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

## Reference Data

The `unit` tests take very little time.

For `integration tests`: 100-structure subsets, Ntrain=40, tested on 2026-07

| example | nconf | Ntrain | % RMSE | threshold | time (1 core) |
|---|---|---|---|---|---|
| water_monomer_aims | 100 | 40 | 9.680e-01 | 1.5 | ~ 4 min |
| water_monomer_pyscf | 100 | 40 | 7.990e-01 | 1.5 | ~ 4 min |
| water_monomer_cp2k | 100 | 40 | 1.497e+00 | 2.5 | ~ 6 min |
| water_monomer_aims (MPI) | 100 | 40 | 9.680e-01 | 1.5 | ~ 6 min |
