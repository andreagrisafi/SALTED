Generate QM/MM training data using CP2K
---------------------------------------
In what follows, we describe how to generate QM/MM training electron densities of a dataset made of Au(100) slabs that interact with a classical Gaussian charge, using the CP2K simulation program.

1. First, set the input argument :code:`inp.qm.qmcode` as :code:`cp2k` and define the path where the CP2K     calculations are going to be perfomed in :code:`inp.qm.path2qm`. Set the periodicity of the system in    :code:`inp.qm.periodic`, choosing among :code:`0D,2D,3D`. Then, initialize the input cell and coordinates    for the CP2K calculation by running:
  
  :code:`python3 -m salted.cp2k.xyz2sys`

2. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file that can be used to do so can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`. 

3. An uncontracted version of this basis can be produced to increase the efficiency of the RI printing workflow, by running:

  :code:`python3 -m salted.cp2k.uncontract_ri_basis contracted_basis_file uncontracted_basis_file`
