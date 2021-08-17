Symmetry-Adapted Learning of Three-dimensional Electron Densities
=================================================================
This repository contains an implementation of sparse Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, that is expanded on an atom-centered basis made of radial functions and spherical harmonics. 


References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)

3. Alan M. Lewis, Andrea Grisafi, Michele Ceriotti, Mariana Rossi, "Learning electron densities in the condensed-phase", arXiv:2106.05364

Installation
-----------------------------
To install it :code:`source env.sh`  

Dependencies
------------
TENSOAP: https://github.com/dilkins/TENSOAP

TENSOAP is set up as a submodule of SALTED. To get the program run :code:`git submodule update --init` in the main folder. Then, go into the TENSOAP folder and install the package.

Input Dataset
-------------
Geometries of the input structures are required in :code:`xyz` format.

The training target consists in the projection of the scalar-field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Given the basis is non-orthogonal, the overlap matrix between the basis functions is also required as an input. 

For each dataset configuration, both the scalar-field projection vector and the overlap matrix are needed. The dimensionality of these arrays has to correspond to the number of atoms, as sorted in the geometry file, times the non-redundant number of basis functions belonging to each atom. The ordering of the basis set follows a hierarchical structure: 

1) For a given atomic species X, loop over the angular orders {L} 

2) For a given combination (X,L), loop over the radial functions {n} 

3) Loop over the angular functions sorted as -L,...,0,...,+L

The possible basis set choices appear in :code:`src/basis.py`. If you want to use a basis that is not included in this file, it is easy enough to add a new one together with the proper dimensions.


a) SALTED water molecules
--------------------------
In this example, we consider the regression of the electron density of a dataset made of 1000 isolated water molecules. For that, go into the example folder :code:`examples/water_monomer`. The file :code:`inp.py` contains the input parameters of the calculation, while the file :code:`water_monomers_1k.xyz` contains the atomic coordinates of the system. Both the file-name of the input geometry and an ordered list of the atomic species included in the dataset need to be specified in :code:`inp.py`. The path to the folder used to save the QM data (overlaps, density coefficients and density projections) must be set using the :code:`path2qm` variable; the path to the folder used to save the ML data produced by SALTED (descriptors, kernels, etc...) must be set using the :code:`path2ml` variable. In the following, we consider the possibility of generating the input densities from scratch. If you want to use your own density matrices, then jump to point 2. If you want to use your own RI-density projections and overlaps, then jump to point 3. 

1) We start generating the density matrices associated with a KS-DFT calculation using PySCF. The QM variables needed as input are the DFT functional (:code:`functional = "b3lyp"`) and the wave-function basis set (:code:`qmbasis = "cc-pvqz"`). To run the QM calculations for each structure in the dataset:: 

        for i in {1..1000}; do python $SALTEDPATH/run_pyscf.py -iconf ${i}; done 

   The density matrices are saved as :code:`path2qm/density_matrices/dm_conf#.npy`.

2) From the density matrices, the resolution of the identity (RI) method can be used to compute the density components on a linear auxiliary basis. The density matrix is assumed to follow the PySCF convention, that is, basis functions are ordered as -L,...,0,...,+L for L>1 and as +1,-1,0 for L=1. The RI (aka density-fitting) auxiliary basis must correspond to its wave-function counterpart and can be set as :code:`dfbasis = "RI-cc-pvqz"`. To compute the RI density coefficients, projections and overlaps for each structure of the dataset, run::

       for i in {1..1000}; do python $SALTEDPATH/dm2df.py -iconf ${i}; done

   These are saved as :code:`path2qm/projections/projections_conf#.npy`, :code:`path2qm/coefficients/coefficients_conf#.npy` and :code:`path2indata/overlaps/overlap_conf#.npy`, respectively.   

3) Using the TENSOAP package, we then need to generate the λ-SOAP structural features up to the maximum angular momentum :code:`-lm` included in the expansion of the density field (up to λ=5 in this case). To do so, run:: 

        cd $path2ml
        mkdir soaps
        cd -

        for i in 0 1 2 3 4 5
        do      
           sagpr_get_PS -f coords_water_monomers_1k.xyz -lm ${i} -s H O -l 4 -n 5 -o $path2ml/soaps/SOAP-${i}
        done 

5) Extract a sparse set of atomic environments to reduce the dimensionality of the regression problem. The number of these environments is specified by the input variable :code:`Menv = 100`. This is done via the farthest point sampling (FPS) method, using the 0-SOAP features previously computed as a metric to distiguish between any pair of atomic environments::

        python $SALTEDPATH/sparse_set.py 


5) Compute the block diagonal kernel matrix for the selected sparse set of atomic environments::  

        python $SALTEDPATH/kernel_mm.py 

6) For each configuration of the dataset, compute the kernel matrix that couples the atoms of that configuration with the selected sparse set of atomic environments:: 

        python $SALTEDPATH/kernel_nm.py

   The kernel matrices can be found in the folder :code:`path2data/kernels/`. 

7) Partition the dataset into training and validation set by selecting :code:`Ntrain = 500` training configurations at random. Then, compute the regression vector A and the regression matrix B using a given training set fraction :code:`trainfrac = 1.0`::

        python $SALTEDPATH/matrices.py 

8) Perform the regression with a given regularization :code:`regul = 1e-08` and jitter value :code:`jitter = 1e-10`, needed to stabilize the matrix inversion::

        python $SALTEDPATH/learn.py 

9) Predict the expansion coefficients of the scalar field over the validation set::

        python $SALTEDPATH/validate.py 
   
   which will be saved as :code:`pred_coeffs.npy`.

10) Print out the predicted scalar field projections in the folder :code:`path2data/predictions/` and compute the root mean square error both on the individual scalar fields (:code:`errors.dat`) and on the overall test dataset (printed out to screen):: 

        python $SALTEDPATH/error_validation.py


    This gives a RMSE of about 0.2% of the intrinsic variability of the electron density over the test set.

11) On top of the predicted density components, compute the Hartree energy and the external energy of the system compared against the RI reference values::

        python $SALTEDPATH/electrostatics.py


    This gives a RMSE of about 0.2 kcal/mol on the final electrostatic energy, corresponding to about 0.03% of the standard deviation over the validation set.


b) ED of water dimers from SALTED water molecules
----------------------------------------------
In this example, we will predict the electron density of 10 water dimers at a large reciprocal distance based on the SALTED exercise carried out for the dataset of isolated water molecules. The input file specifies the file-name of the reference (:code:`water_monomers_1k.xyz`) and new geometry (:code:`water_dimers_10.xyz`), together with the path to the folder where the SALTED exercise has been carried out (:code:`path2ref = ../water_monomer`). Please also specify the path that you used to save the heavy reference data (:code:`path2data_ref`) and the path that you will use to save the new heavy data (:code:`path2data`). If the error associated with the predictions is calculated, the overlaps and projections should be stored at `path2indata`.

Before starting, you need to: i) generate the reference RI-overlaps and RI-density projections of the 10 water dimers as described in points 1)-2) of the previous example, using the very same basis already adopted for the water molecules, ii) compute the L-SOAP features as described in point 4) of the previous example, using the very same parameters adopted for the isolated molecules. The new steps to be undertaken are then described as follows:

1) Compute the cross kernel between the monomers and dimers features::

        python $SALTEDPATH/kernel_tm.py

   The kernel matrices can be found in the folder :code:`path2data/kernels/`.

2) Predict the water dimers densities combining the kernels so computed with the regression weights obtained during the previous example (:code:`path2ref/weights.npy`)::

        python $SALTEDPATH/predict.py

3) Compute the error associated with the predictions::

        python $SALTEDPATH/error_prediction.py

   This gives a RMSE of about 0.2%, according to the isolated molecule case.


Contact
-------
andrea.grisafi@epfl.ch


Contributors
------------
Andrea Grisafi, Alan Lewis, Alberto Fabrizio, Clemence Corminboeuf, Mariana Rossi, Michele Ceriotti
