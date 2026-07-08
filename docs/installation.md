# Installation

!!! warning "Linux only 🐧"
    SALTED is only available on Linux OS. For Windows users, please use WSL or virtual machines.

## Install SALTED

### Quick setup (serial version)

You can find the SALTED program on [GitHub](https://github.com/andreagrisafi/SALTED). In the SALTED directory, run `pip install .` for a serial-only installation.

### Setup parallel version

To use MPI parallelisation you need a parallel (MPI-enabled) h5py, either by `pip` or by `conda`:

- **With pip**: requires a parallel HDF5 and an MPI compiler already available on your system:
    ```bash
    pip install mpi4py
    # set CC to your MPI C compiler (e.g. mpicc)
    HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py
    ```
- **With conda** — provides a parallel HDF5, no compiler needed:
    ```bash
    conda env create -f environment.yml   # add `-n <env-name>` to choose the name
    conda activate salted
    ```


## Install electronic-structure codes

SALTED is to date interfaced with the following electronic-structure codes: *CP2K*, *PySCF*, and *FHI-aims*. If you are interested in using SALTED in combination with other codes, please contact one of the developers.

### PySCF

To install PySCF, you can follow the instructions [here](https://pyscf.org/install.html).

Please note that PySCF works well with small systems like molecules and clusters, but it lacks the scalability to handle periodic systems.
We suggest using CP2K or FHI-aims for these applications.


### FHI-aims


Please use recent versions of FHI-aims, the tutorial presented in this documentation will use the version `240403`.

To install FHI-aims on your cluster or PC, you will need a FHI-aims licence and you can find further information [here](https://fhi-aims.org/get-the-code).
Then you can follow the tutorial [Basics of Running FHI-aims](https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/preparations/) to install FHI-aims.
The `CMake` file is important and you can find more information in the [CMake Tutorial for Compiling FHI-aims (parallel version)](https://aims-git.rz-berlin.mpg.de/aims/FHIaims/-/wikis/CMake%20Tutorial).

Especially, you can find an FHI-aims focused tutorial on SALTED [here in FHI-aims-club](https://fhi-aims-club.gitlab.io/tutorials/fhi-aims-with-salted).

### CP2K

Printing of RI density coefficients and 2-center auxiliary integrals needed to train SALTED is made available starting from the v2023.1 release of CP2K.
