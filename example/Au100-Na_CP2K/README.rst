Generate QM/MM training data using CP2K
---------------------------------------
In what follows, we describe how to generate QM/MM training electron densities of a dataset made of Au(100) slabs that interact with a classical Gaussian charge, using the CP2K simulation program.

1. The following input arguments must be added to the :code:`inp.qm` section:

    - :code:`inp.qm.qmcode`: define quantum-mechanical code as :code:`cp2k`

    - :code:`inp.qm.path2qm`: set the path where the CP2K calculations are going to be perfomed 

    - :code:`inp.qm.periodic`: set the periodicity of the system (:code:`0D,2D,3D`)

2. Initialize the systems used for the CP2K calculation by running:

    :code:`python3 -m salted.cp2k.xyz2sys`

    System cells and coordinates will be automatically saved in folders named :code:`conf_1`, :code:`conf_2`, ... up to the total number of structures included in the dataset, located into the selected :code:`inp.qm.path2qm`. 

2. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`. 

3. An uncontracted version of this basis can be produced to increase the efficiency of the RI printing workflow, by running:

    :code:`python3 -m salted.cp2k.uncontract_ri_basis contracted_basis_file uncontracted_basis_file`

    Then, copy :code:`uncontracted_basis_file` to the :code:`cp2k/data/` folder in order to use this basis set to produce the reference density-fitting data, and set the corresponding filename in the :code:`inp.qm.dfbasis` input variable.

4. Add the selected auxiliary basis to SALTED by running:

    :code:`python3 -m salted.get_basis_info`

5. Run the CP2K calculations and print out the training data made of reference RI coefficients and 2-center auxialiary integrals. An example of a CP2K input file can be found in :code:`cp2k-inputs/qmmm_RI-print.inp`. 

