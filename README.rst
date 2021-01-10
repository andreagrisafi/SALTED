Symmetry-Adapted Learning of Three-dimensional Electron Densities
=================================================================
This repository contains an implementation of sparse Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, that is expanded on an atom-centered basis made of radial functions and spherical harmonics. 


References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)


Requirements and Installation
-----------------------------
This code is written in a mixture of Python2 and Fortran90 with OpenMP parallelization.

To install it, :code:`make` in the main folder and :code:`source env.sh`  


Dependencies
------------
TENSOAP: https://github.com/dilkins/TENSOAP


Input Dataset
-------------
Geometries of the input structures are required in :code:`xyz` format.

The training target consists in the projection of the scalar-field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Given the basis is non-orthogonal, the overlap matrix between the basis functions is also required as an input. Note that the well-conditioning of this matrix is a crucial aspect for the method performance.

For each dataset configuration, both the scalar-field projection vector and the overlap matrix are needed. The dimensionality of these arrays has to correspond to the number of atoms, as sorted in the geometry file, times the non-redundant number of basis functions belonging to each atom. The ordering of the basis set follows a hierarchical structure: 

1) For a given atomic species X, loop over the angular orders {L} 

2) For a given combination (X,L), loop over the radial functions {n} 

3) Loop over the angular functions sorted as -L,...,0,...,+L

The possible basis set choices appear in :code:`src/basis.py`. If you want to use a basis that is not included in this file, it is easy enough to add a new one together with the proper dimensions.


Regression workflow 
-------------------
In this example, we consider the interpolation of the electron density of a dataset made of 1000 water molecules. For that, go into the example folder :code:`examples/water_monomer`. The file :code:`inp.py` contains the input parameters of the calculation, while the file :code:`coords_1000.xyz` contains the atomic coordinates of the system. Both the file-name of the input geometry and an ordered list of the atomic species included in the dataset need to be specified in :code:`inp.py`. In the following, we consider the possibility of generating the input densities from scratch. In case you want to go straight to the regression, you can jump to point 3. 

1) We start generating the density matrices associated with a KS-DFT calculation using PySCF. The QM variables needed as input are the path to the directory where the density matrices will be saved (:code:`path2qm`), the DFT functional (:code:`functional = "b3lyp"`) and the wave-function basis set (:code:`qmbasis = "cc-pvqz"`). To run the QM calculations for each structure in the dataset:: 

        for i in {1..1000}; do python $SALTEDPATH/run_pyscf.py -iconf ${i}; done 

   The density matrices, saved as :code:`dm_conf#.npy`, can be found in the directory specified.

2) From the density matrices, the resolution of the identity (RI) method can be used to compute the density components on a linear auxiliary basis. The density matrix is assumed to be saved according to the PySCF convention, that is, as -L,...,0,...,+L for L>1 and as +1,-1,0 for L=1. The RI-auxiliary basis has to correspond to its wave-function counterpart and can be set as :code:`dfbasis = "RI-cc-pvqz"`. The path to the folders where the RI-density projections and the RI-overlap matrices will be saved can be specified using the variables :code:`path2projs` and :code:`path2overl`, respectively. To compute the RI projections and overlaps for each structure of the dataset, run::

       for i in {1..1000}; do python $SALTEDPATH/dm2df.py -iconf ${i}; done

   The projections and overlaps, saved as :code:`projections_conf#.npy` and :code:`overlap_conf#.npy`, can be found in the directory specified.   

3) In case you skipped points 1 and 2, you can find precomputed projection vectors and the overlap matrices at the RI-cc-pvqz level in the :code:`./projections/` and :code:`./overlaps/` folders. To initialize the regression targets, run::

       python $SALTEDPATH/initialize.py

   This will compute the mean spherical density projections over the dataset and use them as a baseline value for the density projections. 

4) Using the TENSOAP package, generate the L-SOAP structural features up to the maximum angular momentum :code:`-lm` included in the expansion of the density field. In this case, we need to go up to L=5:: 

        for i in 0 1 2 3 4 5
        do      
           sagpr_get_PS -f coords_1000.xyz -lm ${i} -o path2soap/SOAP-${i}
        done 

   Once computed, the path to the folder that you used to save the L-SOAP features need to be specified in :code:`inp.py` using the :code:`path2soap` variable. 

5) Extract a sparse set of atomic environments to reduce the dimensionality of the regression problem. The number of these environments is specified by the input variable :code:`Menv = 100`. This is done via the farthest point sampling (FPS) method, using the 0-SOAP features previously computed as a metric to distiguish between any pair of atomic environments::

        python $SALTEDPATH/sparse_set.py 


5) Compute the block diagonal kernel matrix for the selected sparse set of atomic environments::  

        python $SALTEDPATH/kernel_mm.py 

6) For each configuration of the dataset, compute the kernel matrix that couples the atoms of that configuration with the selected sparse set of atomic environments. The path to the folder used to save the kernels of each configuration needs to be set using the :code:`path2kern` variable. Then run:: 

        python $SALTEDPATH/kernel_nm.py 

7) Partition the dataset into training and validation set by selecting :code:`Ntrain = 500` training configurations at random. Then, compute the regression vector A and the regression matrix B using a given training set fraction :code:`trainfrac = 1.0`::

        python $SALTEDPATH/matrices.py 

8) Perform the regression with a given regularization :code:`regul = 1e-08` and jitter value :code:`jitter = 1e-10`, needed for the stabilize of the solution::

        python $SALTEDPATH/learn.py 

9) Predict the baselined expansion coefficients of the scalar field over the validation set::

        python $SALTEDPATH/validate.py 
   
   which will be saved as :code:`pred_coeffs.npy`.

10) Print out the predicted scalar field projections in the folder specified using the :code:`path2preds` variable and compute the root mean square error both on the individual scalar fields (:code:`errors.dat`) and on the overall test dataset (printed out to screen):: 

        python $SALTEDPATH/error_validation.py


   This gives a RMSE of about 0.2% of the intrisic variability of the electron density over the test set.

11) On top of the predicted density components, compute the Hartree energy and the external energy of the system compared against the RI reference values::

        python $SALTEDPATH/electrostatics.py


   This gives a RMSE of about 0.2 kcal/mol on the final electrostatic energy, corresponding to about 0.03% of the standard deviation over the validation set.


Contact
-------
andrea.grisafi@epfl.ch


Contributors
------------
Andrea Grisafi, Alberto Fabrizio, Alan Lewis, Mariana Rossi, Clemence Corminboeuf, Michele Ceriotti
