# Contributing to SALTED

Thanks for your interest in contributing to SALTED! This document explains how to report issues,
propose changes, and get support.

## Reporting issues or problems

If you encounter a bug, unexpected behavior, or a problem with the documentation:

1. Check the [existing issues](https://github.com/andreagrisafi/SALTED/issues) to see if it has
   already been reported.
2. If not, [open a new issue](https://github.com/andreagrisafi/SALTED/issues/new) including:
   - The version of SALTED you are using (`pip show salted`) and your Python version.
   - The electronic-structure code involved (FHI-aims, CP2K, PySCF), if relevant.
   - Your `inp.yaml` (or the relevant section of it).
   - Steps to reproduce the problem, and the full error traceback if there is one.

## Seeking support

For questions about usage that aren't bug reports, please:

- First check the [documentation](https://salted.readthedocs.io/en/), which includes a
  quick-start guide, worked examples for each interfaced code, and a description of the theory
  behind SALTED.
- Open a [GitHub issue](https://github.com/andreagrisafi/SALTED/issues/new) with the `question`
  label, so the discussion is visible to other users who may have the same question.
- For matters not suited to a public issue, you can reach the maintainers directly (see the
  Contact section of the README).

## Contributing changes

Contributions are welcome, whether they are bug fixes, new features, additional
electronic-structure code interfaces, documentation improvements, or new examples.

### Getting set up

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/SALTED.git
   cd SALTED
   ```

2. Install SALTED in editable mode with the test extras:

   ```bash
   pip install -e ".[test]"
   ```

3. Create a branch for your change:

   ```bash
   git checkout -b my-feature
   ```

### Making changes

- Keep changes focused: a pull request should ideally address one issue or add one feature.
- Add or update docstrings for any new or modified public function.
- If your change affects the user-facing workflow (new `inp.yaml` options, new CLI behavior),
  please also update the relevant page under `docs/`.
- If you are adding a new electronic-structure code interface, please add a worked example under
  `example/`, following the pattern of the existing `water_monomer_*` examples, and document it
  under `docs/examples/`.

### Running the tests

SALTED has two tiers of tests (see `tests/README.md` for details):

- **Unit tests** — fast, no external data required:

  ```bash
  pytest tests/unit -v
  ```

- **Integration tests** — run a full pipeline (training + prediction) against real reference data
  from the companion [SALTED-datasets](https://github.com/andreagrisafi/SALTED-datasets)
  repository, for each interfaced code and for the density-response model:

  ```bash
  git clone --depth 1 https://github.com/andreagrisafi/SALTED-datasets.git
  pytest tests/integration -m aims_density -v -s \
      --require-datasets --datasets-path ./SALTED-datasets
  ```

  Replace `aims_density` with `pyscf_density`, `cp2k_density`, or `aims_response` to run the
  other pipelines. These are the same commands run in CI (`.github/workflows/ci.yaml`).

Please make sure the unit tests pass locally before opening a pull request; if your change
touches a specific pipeline stage or code interface, please also run the corresponding
integration marker.

### Submitting a pull request

1. Push your branch and open a pull request against `master`.
2. Describe what the change does and why, and reference any related issue.
3. GitHub Actions will automatically run the unit tests (and the integration tests, if your PR
   targets `master`). Please make sure these pass, or explain in the PR if a failure is expected
   and unrelated to your change.
4. A maintainer will review your PR and may ask for changes before merging.

By contributing, you agree that your contributions will be licensed under the 
GNU GPLv3+ license.
