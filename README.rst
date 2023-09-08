SALTED: Symmetry-Adapted Learning of Three-dimensional Electron Densities
=========================================================================
This repository contains an implementation of symmetry-adapted Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, decomposed on an atom-centered basis made of radial functions and spherical harmonics. 

References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "Transferable Machine-Learning Model of the Electron Density", *ACS Central Science* **5**, 57 (2019)

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", *Chemical Science* **10**, 9424 (2019)

3. Alan M. Lewis, Andrea Grisafi, Michele Ceriotti, Mariana Rossi, "Learning electron densities in the condensed-phase", *Journal of Chemical Theory and Computation* **17**, 7203 (2021) 

4. Andrea Grisafi, Alan M. Lewis, Mariana Rossi, Michele Ceriotti, "Electronic-Structure Properties from Atom-Centered Predictions of the Electron Density", *Journal of Chemical Theory and Computation* **19**, 4451 (2023) 

Installation
------------
In the SALTED directory, simply run :code:`make`, followed by :code:`pip install .`
   
Dependencies
------------
- **rascaline**: rascaline installation requires a RUST compiler. To install a RUST compiler, run:
:code:`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"`
rascaline can then be installed using
:code:`pip install git+https://github.com/Luthaf/rascaline.git`

- **mpi4py**: mpi4py is required to use MPI parallelisation; SALTED can be run without this.
A parallel h5py installation is required to use MPI parellelisation of equirepr.py only. This can be installed by running:
:code:`HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py`
provided HDF5 has been compiled with MPI support.

Training dataset
----------------
Training data consists in the expansion coefficients of the scalar field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Because of the non-orthogonality of the basis, the overlap matrix between the basis functions is also required as input. The size of these arrays has to correspond to the number of atoms, as sorted in the geometry file, times the number of basis functions belonging to each atom. The ordering of the basis set must follow the structure: 

- For a given atomic species X, cycle over angular momenta L 

- For a given combination (X,L), cycle over radial functions n 

- Cycle over the angular functions sorted as -L,...,0,...,+L

The possible basis set choices appear in :code:`./salted/basis.py` and are consistent with the electronic-structure codes that are to date interfaced with SALTED: **PySCF**, **FHI-aims**, **CP2K**. If you want to use a basis that is not included in this file, follow the code-specific instructions to append the needed information or manually add the proper basis set dimensions to the file.

Usage
-----
For a detailed description of how to use SALTED, refer to the examples corresponding to the electronic structure code you wish to use. SALTED functions may be called either directly from a terminal script, or by importing SALTED modules in python. SALTED input variables must be defined in a :code:`inp.py` file located in the working directory. Input structures are required in XYZ format and are specified in :code:`filename`. SALTED outputs are saved in the directory specified by the input variable :code:`saltedpath`. A general SALTED workflow reads as follows:

- Import SALTED modules

:code:`from salted import equirepr, sparsify, rkhs, feature_vector, matrices, regression, validation`

- Build equivariant structural representations up to the maximum L used to expand the scalar field. 

:code:`equirepr.build()`

- Sparsify equivariant representations over a subset :code:`Menv` of atomic environment and compute RKHS projector as described in Ref.(4). The non-linearity degree of the model can be defined at this stage by setting the zeta parameter :code:`z` as a positive integer. :code:`z=1` corresponds to a linear model.

:code:`sparsify.build()`

- Build equivariant kernels for each density channel and project them over the RKHS as described in Ref.(4).

:code:`rkhs.build()`

- Build SALTED feature vectors for each structure in the training set.

:code:`feature_vector.build()`

- Build regression matrices over a maximum of :code:`Ntrain` training structure. These can be either selected at random :code:'trainsel=``random``' or sequentially :code:'trainsel=``sequential``' from the total dataset. The variable :code:`trainfrac` can be used to define the fraction of the total training data to be used (useful for making learning cruves). 

:code:`matrices.build()`

- Perform regression with a given regularization parameter :code:`regul`.

:code:`regression.build()`

- Validate predictions over the structures that have not been retained for training.

:code:`validation.build()`

Once the SALTED model has been trained and validated, a SALTED prediction on an additional dataset can be performed as follows:

- Import prediction module

:code:`from salted import equipred`

- Perform equivariant prediction

:code:`equipred.build()`

Contact
-------
andrea.grisafi@ens.psl.eu

alan.m.lewis@york.ac.uk

Contributors
------------
Andrea Grisafi, Alan Lewis, Mariana Rossi, Michele Ceriotti
