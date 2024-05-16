# Installation

## About SALTED

The **S**ymmetry-**A**dapted **L**earning of **T**hree-dimensional **E**lectron **D**ensities (SALTED) model was introduced in paper [10.1021/acs.jctc.1c00576  ](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00576), building on previous work for isolated systems [10.1021/acscentsci.8b00551](https://pubs.acs.org/doi/10.1021/acscentsci.8b00551).
The real-space electron density is represented as a linear combination of products of numerical atom-centered orbitals, akin to a density fitting procedure.
The density fitting coefficients are learned from a set of small reference structures using a Symmetry-Adapted Gaussian Process Regression model (SAGPR).
SALTED has been applied to large molecular datasets by enhancing model optimization methods, as detailed in [10.1021/acs.jctc.2c00850  ](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00850).
This tutorial is based on the improved SALTED model presented in this paper.

!!! warning "Linux only 🐧"
    FHI-aims and SALTED are only available on Linux OS. For Windows users, please use WSL or virtual machines.


## Install SALTED

You can find the `SALTED` code [in GitHub](https://github.com/andreagrisafi/SALTED).
To install `SALTED`, please follow the project [README](https://github.com/andreagrisafi/SALTED),
especially the [dependencies](https://github.com/andreagrisafi/SALTED#dependencies) section.

??? note "Editable python package"
    If you want to modify the `SALTED` code, you can install `SALTED` with the following command:

    ```bash
    python -m pip install -e .
    ```

    where `-e` means editable installation, which means you can modify the code and the changes will be reflected in the installed package.
    This is useful for looking into the code / debugging.



## Install an Ab Initio software

SALTED should be installed along with an ab initio software (one of CP2K, PySCF, and FHI-aims). If you develop a different code and would like to use it with SALTED, please contact one of the developers for SALTED.

### PySCF

To install PySCF, you can follow the instructions [here](https://pyscf.org/install.html).

Please note that PySCF works well with small systems like molecules and clusters, but it lacks the scalability to handle periodic systems like crystals.
So we suggest using CP2K or FHI-aims for these applications.


### FHI-aims


In principle please use recent versions of FHI-aims, and for this tutorial we will use the version `240403`.

To install FHI-aims on your cluster or PC, you will need a FHI-aims licence and you can find further information [here](https://fhi-aims.org/get-the-code).
Then you can follow the tutorial [Basics of Running FHI-aims](https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/preparations/) to install FHI-aims.
The `CMake` file is important and you can find more information in the [CMake Tutorial for Compiling FHI-aims (parallel version)](https://aims-git.rz-berlin.mpg.de/aims/FHIaims/-/wikis/CMake%20Tutorial).

Especially, you can find an FHI-aims focused tutorial on SALTED [here in FHI-aims-club](https://fhi-aims-club.gitlab.io/tutorials/fhi-aims-with-salted).


### CP2K

CP2K can be installed from the [CP2K website](https://www.cp2k.org/).


