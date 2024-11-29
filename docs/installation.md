# Installation

!!! warning "Linux only 🐧"
    SALTED is only available on Linux OS. For Windows users, please use WSL or virtual machines.

## Install SALTED

You can find the SALTED program on [GitHub](https://github.com/andreagrisafi/SALTED). In the SALTED directory, simply run `make`, followed by `pip install .`

??? note "Editable python package"
    If you want to modify the code, you can install SALTED with the following command:

    ```bash
    python -m pip install -e .
    ```

    where `-e` means editable installation, which means you can modify the code and the changes will be reflected in the installed package.
    This is useful for looking into the code / debugging.


### Dependencies

 - `rascaline`: rascaline installation requires a RUST compiler. To install a RUST compiler, run: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"`. rascaline can then be installed using `pip install git+https://github.com/Luthaf/rascaline.git`.

 - `mpi4py`: mpi4py is required to use MPI parallelisation; SALTED can nonetheless be run without this. A parallel h5py installation is required to use MPI parellelisation. This can be installed by running: `HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py` provided HDF5 has been compiled with MPI support.

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

The possibility of printing density coefficients and the 2-center auxiliary integrals needed to train SALTED, is made available starting from the v2023.1 release of CP2K.

