Machine-learning of three-dimensional scalar fields 
===================================================
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
TENSOAP: <TENSOAP>


Input Dataset
-------------
The geometry of the training configurations have to be stored in :code:`xyz` format.

The training dataset required to run the regression consists in the projection of the scalar-field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the raidal functions, which can be either orthogonal or non-orthogonal to each other depending on the user choice. The overlap matrix between the basis functions is also required as an input. The well-conditioning of this matrix is a crucial aspect for the method performance.

For each dataset configuration, both the scalar-field projection vector and the overlap matrix need to be stored in numpy binary arrays within the folders :code:`projections` and :code:`overlaps` respectively. The dimensionality of these arrays has to correspond to the number of atoms as sorted in the geometry file, times the non-redundant number of basis functions belonging to each atom. The ordering of the basis follows this hierarchical structure: 

1) For a given atomic species S, loop over the possible angular momenta {L}

2) Loop over the possible radial channels {n} associated with a given combination for S and L

3) Finally, loop over the angular momentum components sorted as -L,...,0,...,+L

The possible basis set choices appear in :code:`src/basis.py`. If you want to use a basis that is not included in this file, it is easy enough to add a new one together with the proper dimensions.


Workflow 
--------
In the following, the interpolation of the electron density of a dataset of 1000 water molecules is considered as an example. For that, go into the example folder :code:`examples/water_monomer`. There you will find the file :code:`inp.py`, containing the input parameters of the calculation. 

1) Generate L-SOAP representations up to the maximum angular momentum :code:`-lm` included in the expansion of the scalar field. In this case, we need to go up to L=5:: 

        for i in 0 1 2 3 4 5
        do
           sagpr_get_PS -n 8 -l 6 -rc 4.0 -sg 0.3 -f coords_1000.xyz -c H O -s H O -lm ${i} -o SOAP-${i}
        done 

   Type :code:`get_power_spectrum.py -h` for SOAP parameters documentation. Note that to sensibly reduce the feature space size for high angular momenta, the resolution of the SOAP representation can possibly be decreased as the :code:`-lm` value is increased, without loosing in learning accuracy. This means reducing the radial :code:`-n` and angular :code:`-l` cutoffs respectively used to expand the SOAP atomic density.

2) Extract a sparse set of environmentsto reduce the dimensionality of the regression problem. This is done via the farthest point sampling (FPS) method, using the SOAP-0 representation previously computed as a metric to distiguish between two atomic environments::

        python $RHOMLPATH/sparse_set.py 


3) Compute the block diagonal kernel matrix for the selected sparse set of atomic environments::  

        python $RHOMLPATH/kernel_mm.py 

4) For each configuration of the dataset, compute the kernel matrix that couples the atoms of that configuration with the selected sparse set of atomic environments. For that, first generate a folder as :code:`mkdir kernels` where the kernel matrices for each configuration will be saved as text files::

        python $RHOMLPATH/kernel_nm.py 

5) Compute the spherical averages of the scalar field coefficients over the dataset and use them to baseline the projections of the scalar field on the basis set. Then print out both the baselined projections and overlap matrices as text files::

        python $RHOMLPATH/initialize.py

6) Partition the dataset into training and test set by selecting N training configurations at random. Then compute the regression vector A and the regression matrix B using a given training set fraction specified in :code:`inp.py`::

        python $RHOMLPATH/matrices.py 

7) Do the regression with a given regularization and jitter values specified in :code:`inp.py`, needed for the stability of the matrix inversion::

        python $RHOMLPATH/learn.py 

8) Predict the baselined expansion coefficients of the scalar field over the test set::

        python $RHOMLPATH/predict.py 

9) Print out the predicted scalar field projections in the :code:`projections` folder and estimate the root mean square error both on the individual scalar fields (:code:`errors.dat`) and on the overall test dataset:: 

        python $RHOMLPATH/error.py


Contact
-------
andrea.grisafi@epfl.ch


Contributors
------------
Andrea Grisafi, Alberto Fabrizio, Alan Lewis
