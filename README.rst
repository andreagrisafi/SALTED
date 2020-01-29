Machine-learning of three-dimensional scalar fields 
===================================================

This repository contains an implementation of sparse Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, that is expanded on an atom-centered basis made of radial functions and spherical harmonics. 

References
----------

1. Andrea Grisafi, David M. Wilkins, Gabor Csányi, Michele Ceriotti, "Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic Systems", Physical Review Letters 120, 036002 (2018)

2. Andrea Grisafi, David M. Wilkins, Benjamin A. R. Meyer, Alberto Fabrizio, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

3. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)

Requirements and Installation
-----------------------------
This code is written in a mixture of Python2 and Fortran90 with OpenMP parallelization.

To install it, just :code:`make` in the main folder. 

Workflow 
--------

In the following, the interpolation of the electron density of a dataset of 1000 water molecules is considered as an example. For that, go in the example folder :code:`examples/water_monomer`.

1) Generate lambda-SOAP representations up to the maximum angular momentum :code:`-lm` included in the expansion of the scalar field. For instance, when going up to L=3 spherical harmonics:: 

        for i in 0 1 2 3 4 5
        do
           $path_to_soapfast/SOAPFAST/soapfast/get_power_spectrum.py -n 8 -l 6 -rc 4.0 -sg 0.3 -f coords_1000.xyz -s H O -lm ${i} -o SOAP-${i}
        done 

   Type :code:`get_power_spectrum.py -h` for SOAP parameters documentation. Note that to sensibly reduce the feature space size for high angular momenta, the resolution of the SOAP representation can possibly be made coarser as the lambda :code:`-lm` value is increased, without loosing in learning accuracy.

2) Extract a sparse set of environments :code:`-m` to reduce the dimensionality of the regression problem. This is done via the farthest point sampling (FPS) method, using the SOAP-0 representation previously computed as a metric to distiguish between two atomic environments::

        python ../../src/sparse_set.py -f coords_1000.xyz -m 100


3) Compute the block diagonal kernel matrix for the selected sparse set of atomic environments::  

        python ../../src/kernel_mm.py 

4) For each configuration of the dataset, compute the kernel matrix that couples the atoms of that configuration with the selected sparse set of atomic environments::

        mkdir kernels 
        python ../../src/kernel_nm.py 

5) Compute the spherical averages of the scalar field coefficients over the dataset and use them to baseline the projections of the scalar field on the basis set. Then print out both the baselined projections and overlap matrix into :code:`.dat` files::

        python ../../src/initialize.py

6) Partition the dataset into training and test set by selecting :code:`-t` training configurations at random and compute the regression vector A and the regression matrix B by using a given training set fraction `-frac`::

        python ../../src/matrices.py -t 200 -frac 1.0

7) Do the regression with a given regularization :code:`-r` and jitter value :code:`-j` needed for the stability of the matrix inversion::

        python ../../src/learn.py -r 1e-08 -jit 1e-10

8) Predict the baselined expansion coefficients of the scalar field over the test set::

        python ../../src/predict.py 

9) Print out the predicted scalar field projections in the :code:`projections` folder and estimate the root mean square error both on the individual predicted scalar fields and on the overall test dataset:: 

        python ../../src/error.py
