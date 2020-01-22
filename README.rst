Machine-learning of three-dimensional scalar fields 
===================================================

This repository contains an implementation of sparse Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, that is expanded on an atom-centered basis made of radial functions and spherical harmonics. 

References
----------

1. Andrea Grisafi, David M. Wilkins, Gabor Csányi, Michele Ceriotti, "Symmetry-Adapted Machine Learning for Tensorial Properties of Atomistic Systems", Physical Review Letters 120, 036002 (2018)

2. Andrea Grisafi, David M. Wilkins, Benjamin A. R. Meyer, Alberto Fabrizio, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

3. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, Michele Ceriotti, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)

Requirements and Installation
-----------------------------
This code is written in a mixture of Python2 and Fortran90 with OpenMP parallelization.
To install it, just type :code:`make` in the main folder.

Workflow
--------

1) Generate lambda-SOAP representations up to the maxium angular momentum included in the expansion of the density, e.g.,::

        sagpr_get_PS -n 8 -l 6 -rc 4.0 -sg 0.3 -lm 5 -f coords_1000.xyz -s H O -c H O -o PS_5

2) Define a sparse set of environments :code:`-m` using the FPS method with the 0-SOAP metric::

        python environments.py -m 100

3) Compute the environmental kernel matrices which couple the sparse set with the training set:: 

        python src/kernels.py -m 100

4) Compute the environmental kernel matrices for the sparse set::  

        python src/rmatrix.py -m 100

5) Initialize density projections and and basis set overlaps (depending on the type of input given)::

        python src/initialize.py

6) Compute spherical averages of the density components over the training and use them as baseline values for the density projections::

        python src/baseline_projs.py

7) Compute the regression vector B and the regression matrix A using a given training set fraction `-f`::

        python src/get_matrices.py -m 100 -f 1.0

8) Do the regression with a given regularization :code:`-r` and jitter :code:`-j` needed for the stability of the matrix inversion::

        python src/regression.py -f 1.0 -m 100 -r 1e-06 -jit 1e-08

9) Perform predictions of the density coefficients on the test set::

        python src/prediction.py -f 1.0 -m 100 -r 1e-06 -jit 1e-08

10) Estimate the root mean square error on the predicted scalar field:: 

        python src/compute_error.py -f 1.0 -m 100 -r 1e-06 -jit 1e-08
