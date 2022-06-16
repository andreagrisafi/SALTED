Symmetry-Adapted Learning of Three-dimensional Electron Densities
=================================================================
This repository contains an implementation of sparse Gaussian Process Regression that is suitable to do machine learning of any three-dimensional scalar field, e.g., the electron density of a system, that is expanded on an atom-centered basis made of radial functions and spherical harmonics. 


References
----------
1. Andrea Grisafi, Alberto Fabrizio, David M. Wilkins, Benjamin A. R. Meyer, Clemence Corminboeuf, Michele Ceriotti, "A Transferable Machine-Learning Model of the Electron Density", ACS Central Science 5, 57 (2019)

2. Alberto Fabrizio, Andrea Grisafi, Benjamin A. R. Meyer, Michele Ceriotti, Clemence Corminboeuf, "Electron density learning of non-covalent systems", Chemical Science 10, 9424-9432 (2019)

3. Alan M. Lewis, Andrea Grisafi, Michele Ceriotti, Mariana Rossi, "Learning electron densities in the condensed-phase", Journal of chemical theory and computation 17 (11), 7203-7214 (2021) 

Dependencies
------------
TENSOAP: https://github.com/dilkins/TENSOAP

Input Dataset
-------------
Geometries of the input structures are required in :code:`xyz` format.

The training target consists in the projection of the scalar-field over atom-centered basis functions made of radial functions and spherical harmonics. We assume to work with real spherical harmonics defined with the Condon-Shortley phase convention. No restriction is instead imposed on the nature of the radial functions. Given the basis is non-orthogonal, the overlap matrix between the basis functions is also required as an input. 

For each dataset configuration, both the scalar-field projection vector and the overlap matrix are needed. The dimensionality of these arrays has to correspond to the number of atoms, as sorted in the geometry file, times the non-redundant number of basis functions belonging to each atom. The ordering of the basis set follows a hierarchical structure: 

1) For a given atomic species X, loop over the angular orders {L} 

2) For a given combination (X,L), loop over the radial functions {n} 

3) Loop over the angular functions sorted as -L,...,0,...,+L

The possible basis set choices appear in :code:`src/basis.py`. If you want to use a basis that is not included in this file, it is easy enough to add a new one together with the proper dimensions.

Contact
-------
andrea.grisafi@ens.psl.eu

Contributors
------------
Andrea Grisafi, Alan Lewis, Mariana Rossi, Michele Ceriotti
