# Installation

!!! warning "Windows users"
    SALTED is available on Linux OS and MacOS. Windows users please use WSL or virtual machines.

## Install SALTED

Install SALTED from [PyPI](https://pypi.org/project/salted-gpr/) (Python 3.10 to 3.14; the package is `salted-gpr`, the import name stays `salted`):

```bash
pip install salted-gpr
```

or the development version directly from [GitHub](https://github.com/andreagrisafi/SALTED):

```bash
pip install git+https://github.com/andreagrisafi/SALTED.git
```

### Supported Python versions

SALTED supports **Python 3.10, 3.11, 3.12, 3.13 and 3.14** on Linux and macOS; CI runs the unit tests on each of them.
It is pure Python, so the one `salted-gpr` package serves all of these versions.

What sets the limits:

- **Oldest: 3.10.** SALTED uses the `X | Y` type-union syntax, which needs Python 3.10, and `featomic` 0.6.4 and later require Python 3.10.
  Python 3.9 reached end of life in October 2025, and current releases of NumPy, SciPy and Numba no longer support it.
  Python 2 has never been supported.
- **Newest: 3.14.** Python 3.15 will be added once all dependencies publish packages for it (h5py does not yet).


## Install electronic-structure codes

SALTED is to date interfaced with the following electronic-structure codes: *CP2K*, *PySCF*, and *FHI-aims*. If you are interested in using SALTED in combination with other codes, please contact one of the developers.

### PySCF

PySCF is installed automatically as a core dependency of SALTED.
It is required by PySCF datasets, also some calculations in CP2K workflow.

Please note that PySCF works well with small systems like molecules and clusters, but it lacks the scalability to handle periodic systems.
We suggest using CP2K or FHI-aims for these applications.


### FHI-aims


Please use recent versions of FHI-aims, the tutorial presented in this documentation will use the version `240403`.

To install FHI-aims on your cluster or PC, you will need a FHI-aims licence and you can find further information [here](https://fhi-aims.org/get-the-code).
Then you can follow the tutorial [Basics of Running FHI-aims](https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/preparations/) to install FHI-aims.
The `CMake` file is important and you can find more information in the [CMake Tutorial for Compiling FHI-aims (parallel version)](https://aims-git.rz-berlin.mpg.de/aims/FHIaims/-/wikis/CMake%20Tutorial).

Especially, you can find an FHI-aims focused tutorial on SALTED [here in FHI-aims-club](https://fhi-aims-club.gitlab.io/tutorials/fhi-aims-with-salted).

### CP2K

**CP2K v2026.2 or later** is required. Please refer to the [CP2K documentation](https://manual.cp2k.org/trunk/index.html) to download and install CP2K.
