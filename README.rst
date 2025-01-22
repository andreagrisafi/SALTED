SALTED: Symmetry-Adapted Learning of Three-dimensional Electron Densities
=========================================================================
This repository contains an implementation of symmetry-adapted Gaussian Process Regression suitable to perform equivariant learning and prediction of the electron density of molecular and condensed-phase systems, together with its static linear response function to applied electric fields. This is done by representing the continuous scalar (density) and vector (density-response) fields on a linear basis of atom-centered radial functions and spherical harmonics.

Documentation
-------------
A quick-start guide is provided below; `full documentation is also available <https://salted.readthedocs.io/en/>`_.

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

--> **featomic**: featomic installation requires a RUST compiler. To install a RUST compiler, run:
:code:`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"`
featomic can then be installed using
:code:`pip install git+https://github.com/metatensor/featomic.git`

--> **mpi4py**: mpi4py is required to use MPI parallelisation; SALTED can nonetheless be run without this.
A parallel h5py installation is required to use MPI parellelisation. This can be installed by running:
:code:`HDF5_MPI="ON" CC=mpicc pip install --no-cache-dir --no-binary=h5py h5py`
provided HDF5 has been compiled with MPI support.

--> :code: `pip install meson ninja` to run f2py using meson backend following versions of Python > 3.12.

Input file
----------
SALTED input is provided in a :code:`inp.yaml` file, which is structured in the following sections:

- :code:`salted` (required): define root storage directory, workflow label and learning target 

- :code:`system` (required): define system parameters 

- :code:`qm` (required): define information about quantum-mechanical reference

- :code:`descriptor` (required): define parameters of symmetry-adapted descriptors

- :code:`gpr` (required): define Gaussian Process Regression parameters 

- :code:`prediction` (optional): manage predictions on unseen datasets  

Input Dataset
-------------
Input structures are required in extXYZ format; the corresponding filename must be specified in the :code:`inp.system.filename`.  
Training data consists in the expansion coefficients of the scalar/vector field over atom-centered basis functions made of radial functions and spherical harmonics. These coefficients are computed following density-fitting (DF), a.k.a. resolution of the identity, approximations, commonly applied in electronic-structure codes. We assume to work with orthonormalized real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Because of the non-orthogonality of the basis functions, the 2-center electronic integral matrices associated with the given density-fitting approximation are also required as input. 
The electronic-structure codes that are to date interfaced with SALTED are:
    
   - **FHI-aims**
   - **CP2K** 
   - **PySCF** 

We refer to the code-specific examples for how to produce the required quantum-mechanical data. 

Usage
-----
The root directory used for storing SALTED data is specified in :code:`inp.salted.saltedpath`. Depending on the chosen input parameters, a SALTED workflow can be labelled adding a coherent string in the :code:`inp.salted.saltedname` variable; in turn, this defines the name of the output folders that are automatically generated during the program execution. The type of SALTED target can be selected by specifying :code:`inp.salted.saltedtype: density`, when asking to learn electron density, or :code:`inp.salted.saltedtype: density-response`, when asking to learn the electron-density linear response to applied electric fields. SALTED functions can be run either by importing the corresponding modules in Python, or directly from command line. 
MPI parallelization can be activated by setting :code:`inp.system.parallel` as :code:`True`, and can be used, whenever applicable, to parallelize the calculation of SALTED functions over training data. 
In what follows, we report an example of a general command line workflow: 

1. Initialize structural features defined from 3-body symmetry-adapted descriptors, $P^L$, as computed following PRL 120, 036002 (2018):

   :code:`python3 -m salted.initialize`

   An optional :code:`sparsify` subsection can be added to the :code:`inp.descriptor` input section in order to reduce the feature space size down to :code:`ncut` sparse features selected using a "farthest point sampling" (FPS) algorithm. To facilitate this procedure, it is possible to perform the FPS selection over a subset of :code:`nsamples` configurations, selected at random from the entire training dataset.

2. Find sparse set of :code:`inp.gpr.Menv` atomic environments in order to recast the SALTED problem into a low dimensional space. The non-linearity degree of the model must be defined at this stage by setting the variable :code:`inp.gpr.z` as a positive integer. :code:`z=1` corresponds to a linear model. 

   :code:`python3 -m salted.sparse_selection`

3. Compute sparse vectors of descriptors $P^L_M$ for each atomic type and angular momentum: 

   :code:`python3 -m salted.sparse_descriptor` (MPI parallelizable)

4. Compute sparse equivariant kernels $k^L_{MM}$ and find projector matrices over the Reproducing Kernel Hilbert Space (RKHS):

   :code:`python3 -m salted.rkhs_projector`

5. Compute equivariant kernels $k^L_{NM}$ over the entire dataset and project them on the RKHS to obtain the final SALTED input vectors: 

   :code:`python3 -m salted.rkhs_vector` (MPI parallelizable)

6. Build the Hessian matrix of the quadratic RKHS problem over a maximum of :code:`inp.gpr.Ntrain` training structures selected from the entire dataset; these can be either selected at random (:code:`inp.gpr.trainsel: random`) or sequentially (:code:`inp.gpr.trainsel: sequential`). The remaining structures will be automatically retained for validation.  The variable :code:`inp.gpr.trainfrac` can be used to define the fraction of the total training data to be used: this can go from 0 to 1 in order to make learning curves while keeping the validation set fixed. 

   :code:`python3 -m salted.hessian_matrix` (MPI parallelizable)

7. Solve the regression problem with a given regularization parameter :code:`inp.gpr.regul`. 

   :code:`python3 -m salted.solve_regression`

   NB: when the dimensionality exceeds $10^5$, it is recommended to perform a direct minimization of the SALTED loss function in place of an explicit matrix inversion (points 6 and 7). If the dimensionality exceeds $10^5$, the loss function must be minimized directly. This can be run as follows:

   :code:`python3 -m salted.minimize_loss` (MPI parallelizable)

8. Validate predictions over the structures that have not been retained for training by computing the root mean square error in agreement to the definition of the SALTED loss function.

   :code:`python3 -m salted.validation` (MPI parallelizable)

9. Once the SALTED model has been trained and validated, SALTED predictions for a new unseen dataset can be handled according to the :code:`inp.prediction` section. For that, a :code:`inp.prediction.filename` must be specified in XYZ format, while a :code:`inp.prediction.predname` string can be defined to label the prediction directories. Equivariant predictions can then be run as follows:

   :code:`python3 -m salted.prediction` (MPI parallelizable) 

Contact
-------
andrea.grisafi@ens.psl.eu

alan.m.lewis@york.ac.uk

