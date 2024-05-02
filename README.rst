SALTED: Symmetry-Adapted Learning of Three-dimensional Electron Densities
=========================================================================
This repository contains an implementation of symmetry-adapted Gaussian Process Regression that is suitable to do machine learning of the electron density of a system, decomposed on an atom-centered spherical harmonics basis. 

References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "Transferable Machine-Learning Model of the Electron Density", *ACS Central Science* **5**, 57 (2019) [https://pubs.acs.org/doi/10.1021/acscentsci.8b00551]

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", *Chemical Science* **10**, 9424 (2019) [https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc02696g]

3. Alan M. Lewis, Andrea Grisafi, Michele Ceriotti, Mariana Rossi, "Learning electron densities in the condensed-phase", *Journal of Chemical Theory and Computation* **17**, 7203 (2021) [https://pubs.acs.org/doi/10.1021/acs.jctc.1c00576]

4. Andrea Grisafi, Alan M. Lewis, Mariana Rossi, Michele Ceriotti, "Electronic-Structure Properties from Atom-Centered Predictions of the Electron Density", *Journal of Chemical Theory and Computation* **19**, 4451 (2023) [https://pubs.acs.org/doi/10.1021/acs.jctc.2c00850]

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
SALTED input is provided in a :code:`inp.yaml` file, which is structured in the following sections:

- :code:`salted` (required): define root storage directory and workflow label 

- :code:`system` (required): define system parameters 

- :code:`qm` (required): define information about quantum-mechanical reference

- :code:`descriptor` (required): define parameters of symmetry-adapted descriptors

- :code:`gpr` (required): define Gaussian Process Regression parameters 

- :code:`prediction` (optional): manage predictions on unseen datasets  

Input Dataset
-------------
Input structures are required in XYZ format; the corresponding filename must be specified in the :code:`inp.system.filename`. 
Electron density training data consists in the expansion coefficients of the scalar field over atom-centered basis functions made of radial functions and spherical harmonics. These coefficients are computed following density-fitting (DF), a.k.a. resolution of the identity, approximations, commonly applied in electronic-structure codes. We assume to work with orthonormalized real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Because of the non-orthogonality of the basis functions, the 2-center electronic integral matrices associated with the given density-fitting approximation are also required as input. 
The electronic-structure codes that are to date interfaced with SALTED are **FHI-aims**, **CP2K** and **PySCF**; we refer to the code-specific examples for how to produce the required quantum-mechanical data. The selected DF basis must be specified into the :code:`inp.qm.dfbasis` input argument, and then added to SALTED through the :code:`salted.get_basis_info` function.

Usage
-----
The root directory used for storing SALTED data is specified in :code:`inp.salted.saltedpath`. Depending on the chosen input parameters, a SALTED workflow can be labelled adding a coherent string in the :code:`inp.salted.saltedname` variable; in turn, this defines the name of the output folders that are automatically generated during the program execution. SALTED functions can be run either by importing the corresponding modules in Python, or directly from command line. 
MPI parallelization can be activated by setting :code:`inp.system.parallel` as :code:`True`, and can be used, whenever applicable, to parallelize the calculation of SALTED functions over training data. 
In what follows, we report an example of a general command line workflow: 

- Initialize structural features defined from 3-body symmetry-adapted descriptors, as computed following PRL 120, 036002 (2018):

   :code:`python3 -m salted.init_features`

   An optional :code:`sparsify` subsection can be added to the :code:`inp.descriptor` input section in order to reduce the feature space size down to :code:`ncut` sparse features selected using a "farthest point sampling" (FPS) algorithm. To facilitate this procedure, it is possible to perform the FPS selection over a subset of :code:`nsamples` configurations, selected at random from the entire training dataset.

- Find sparse set of :code:`inp.gpr.Menv` atomic environments in order to recast the SALTED problem on a low dimensional space. The non-linearity degree of the model must be defined at this stage by setting the variable :code:`inp.gpr.z` as a positive integer. :code:`z=1` corresponds to a linear model. 

   :code:`python3 -m salted.sparse_selection`

- Compute sparse vectors of descriptors for each atomic type and angular momentum: 

   :code:`python3 -m salted.sparse_vector` (MPI parallelizable)

- Compute sparse equivariant kernels and find projector matrices over the Reproducing Kernel Hilbert Space (RKHS) as described in Ref.(4):

   :code:`python3 -m salted.rkhs_projector`

- Compute equivariant kernels and project them on the RKHS to obtain the final SALTED input vectors: 

   :code:`python3 -m salted.rkhs_vector` (MPI parallelizable)

- Build regression matrices over a maximum of :code:`inp.gpr.Ntrain` training structure. These can be either selected at random (:code:`inp.gpr.trainsel="random")` or sequentially (:code:`inp.gpr.trainsel="sequential"`) from the entire dataset. The variable :code:`inp.gpr.trainfrac` can be used to define the fraction of the total training data to be used (useful for making learning cruves). 

   :code:`python3 -m salted.matrices` (MPI parallelizable)

- Perform regression with a given regularization parameter :code:`inp.gpr.regul`. 

   :code:`python3 -m salted.regression`

- When the dimensionality of the learning problem exceeds $10^5$, it is recommended to perform an explicit minimization of the SALTED loss function:

   :code:`python3 -m salted.minimize_loss` (MPI parallelizable)

- Validate predictions over the structures that have not been retained for training by computing the root mean square error in agreement to the definition of the SALTED loss function.

   :code:`python3 -m salted.validation` (MPI parallelizable)

Once the SALTED model has been trained and validated, SALTED predictions for a new unseen dataset can be handled according to the :code:`inp.prediction` section. For that, a :code:`inp.prediction.filename` must be specified in XYZ format, while a :code:`inp.prediction.predname` string can be defined to label the prediction directories. Equivariant predictions over a given thataset can then be run as follows:

   :code:`python3 -m salted.equipred` (MPI parallelizable) 

Contact
-------
andrea.grisafi@ens.psl.eu

alan.m.lewis@york.ac.uk

Contributors
------------
Andrea Grisafi, Alan Lewis, Zekun Lou
