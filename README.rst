SALTED: Symmetry-Adapted Learning of Three-dimensional Electron Densities
=========================================================================
This repository contains an implementation of symmetry-adapted Gaussian Process Regression that is suitable to do machine learning of the electron density of a system, decomposed on an atom-centered spherical harmonics basis. 

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

- **mpi4py**: mpi4py is required to use MPI parallelisation; SALTED can nonetheless be run without this.
A parallel h5py installation is required to use MPI parellelisation. This can be installed by running:
:code:`HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py`
provided HDF5 has been compiled with MPI support.

Input file
----------
SALTED input is provided in a :code:`inp.yaml` file. 

Input Dataset
-------------
Input structures are required in XYZ format; the corresponding filename must be specified in the :code:`inp.system.filename` input argument. 
Training data consists in the expansion coefficients of the scalar field over atom-centered basis functions made of radial functions and spherical harmonics. These coefficients are computed following density-fitting (DF), a.k.a. resolution of the identity, approximations, commonly applied in electronic-structure codes. We assume to work with orthonormalized real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Because of the non-orthogonality of the basis functions, the 2-center electronic integral matrices associated with the given density-fitting approximation are also required as input. 
The electronic-structure codes that are to date interfaced with SALTED are **FHI-aims**, **CP2K** and **PySCF**; we refer to the code-specific examples for how to produce the required quantum-mechanical data. Once the data are produced, you can add the selected DF basis into the :code:`inp.qm.dfbasis` input argument and run the :code:`salted.get_basis_info` function to add this to SALTED.

Usage
-----
SALTED input and output folders are saved in the directory specified by the input variable :code:`saltedpath`. In order to test different SALTED models, a :code:`saltedname` string must also be specified which will be appended to the name of the output folders that are automatically generated during the program execution. A general SALTED workflow reads as follows:

- Import SALTED modules

:code:`from salted import equirepr, sparsify, rkhs, feature_vector, matrices, regression, validation`

- Build symmetry-adapted representations of the atomic structure, one for each angular momentum used to expand the electron density. These are constructed as 3-body atom-centered descriptors that mirror spherical harmonics transformations in three dimensions, as described in PRL **120**, 036002 (2018). It is possible to sparsify the feature space by setting :code:`sparsify=True` together with the positive integer :code:`nc` in order to retain a corresponding number of features. The sparsification is performed with a "farthest point sampling" (FPS) algorithm using the full descriptor to define the Euclidean distance between the structural features. In order to reduce the computational burden of this procedure, it is possible to perform the FPS selection over a prescribed subset of samples :code:`nsamples` chosen at random from the entire training dataset.

:code:`equirepr.build()`

- Sparsify training set by selecting a subset :code:`Menv` of atomic environments in order to compute sparse kernels and truncated RKHS projectors as described in Ref.(4). The non-linearity degree of the model must be defined at this stage by setting the variable :code:`z` as a positive integer. :code:`z=1` corresponds to a linear model. The truncation of small and negative eigenvalues when computing the RKHS projector can be modulated by the threshold :code:`eigcut` (default value $10^{-10}$).

:code:`sparsify.build()`

- Build equivariant kernels for each density channel and project them over the RKHS as described in Ref.(4).

:code:`rkhs.build()`

- Build SALTED feature vectors for each structure in the training set.

:code:`feature_vector.build()`

- Build regression matrices over a maximum of :code:`Ntrain` training structure. These can be either selected at random :code:`trainsel="random"` or sequentially :code:`trainsel="sequential"` from the total dataset. The variable :code:`trainfrac` can be used to define the fraction of the total training data to be used (useful for making learning cruves). 

:code:`matrices.build()`

- Perform regression with a given regularization parameter :code:`regul`. NB: An explicit minimization of the loss function is recommended when the dimensionality of the problem exceeds $10^5$; see the MPI examples for how to run SALTED in this case.

:code:`regression.build()`

- Validate predictions over the structures that have not been retained for training by estimating the density error as reported in Ref.(4).

:code:`validation.build()`

Once the SALTED model has been trained and validated, SALTED predictions for a new dataset can be performed as follows:

- Import prediction module

:code:`from salted import equipred`

- Perform equivariant predictions and save prediction outputs in dedicated folders located in :code:`saltedpath` by making use of a :code:`predname` string that is appended to the name of the prediction directories.

:code:`equipred.build()`

Contact
-------
andrea.grisafi@ens.psl.eu

alan.m.lewis@york.ac.uk

Contributors
------------
Andrea Grisafi, Alan Lewis
