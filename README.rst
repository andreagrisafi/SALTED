Symmetry-Adapted Learning of Three-dimensional Electron Densities
=================================================================
This repository contains an implementation of symmetry-adapted Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, decomposed on an atom-centered basis made of radial functions and spherical harmonics. 


References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)

3. Alan M. Lewis, Andrea Grisafi, Michele Ceriotti, Mariana Rossi, "Learning electron densities in the condensed-phase", Journal of chemical theory and computation 17 (11), 7203-7214 (2021) 

4. Andrea Grisafi, Alan M. Lewis, Mariana Rossi, Michele Ceriotti,, "Electronic-Structure Properties from Atom-Centered Predictions of the Electron Density", Journal of chemical theory and computation 19 (14), 4451-4460 (2023) 

Installation
------------
In the SALTED directory, simply run :code:`make`, followed by :code:`pip install .`
   
Dependencies
------------
**rascaline**: rascaline installation requires a RUST compiler. To install a RUST compiler, run:
:code:`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"`
rascaline can then be installed using
:code:`pip install git+https://github.com/Luthaf/rascaline.git`

**mpi4py**: mpi4py is required to use MPI parallelisation; SALTED can be run without this.
A parallel h5py installation is required to use MPI parellelisation of equirepr.py only. This can be installed by running:
:code:`HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py`
provided HDF5 has been compiled with MPI support.

Usage
-----
For detailed examples of how to use SALTED, refer to the example corresponding to the electronic structure code you wish to use. In general, functions may be called either directly from a terminal script, or using a python script via :code:`import salted`.

Input Dataset
-------------
Geometries of the input structures are required in :code:`xyz` format.

Training data consists in the projection of the scalar field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. The overlap matrix between the basis functions is also required as an input. The size of these arrays has to correspond to the number of atoms, as sorted in the geometry file, times the non-redundant number of basis functions belonging to each atom. The ordering of the basis set follows a hierarchical structure: 

1) For a given atomic species X, cycle over the angular index {L} 

2) For a given combination (X,L), cycle over the (contracted) radial functions {n} 

3) Cycle over the angular functions sorted as -L, ..., 0 , ... , +L

The possible basis set choices appear in :code:`src/basis.py` and are consistent with the electronic structure codes that are to date interfaced with SALTED, i.e., PySCF, FHI-aims, CP2K. If you want to use a basis that is not included in this file, follow the code-specific instructions to append the needed information or manually add the proper dimensions to the file.

Contact
-------
andrea.grisafi@ens.psl.eu

alan.m.lewis@york.ac.uk

Contributors
------------
Andrea Grisafi, Alan Lewis, Mariana Rossi, Michele Ceriotti
