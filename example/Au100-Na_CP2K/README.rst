Generate QM/MM training data using CP2K
---------------------------------------
In what follows, we describe how to generate QM/MM training electron densities of a dataset made of Au(100) slabs that interact with a classical Gaussian charge, using the CP2K simulation program.

1. Print auxiliary basis set information from the automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file that can be used to do so can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`.
