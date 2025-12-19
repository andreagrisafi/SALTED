Generate training data using CP2K
----------------------------------

In what follows, we describe how to generate training electron densities of a dataset made of Au(100) slabs that interact with a classical Gaussian charge, using the CP2K simulation program. NB: this is made possible through to the official development version of CP2K (https://github.com/cp2k/cp2k).

1. The following input arguments must be added to the :code:`inp.qm` section:

   :code:`qmcode`: define quantum-mechanical code as :code:`cp2k`

   :code:`path2qm`: set the path where the CP2K data are going to be saved

   :code:`periodic`: set the periodicity of the system (:code:`0D,2D,3D`)

   :code:`coeffile`: filename of RI density coefficients as printed by CP2K

   :code:`ovlpfile`: filename of 2-center auxiliary integrals as printed by CP2K

   :code:`dfbasis`: extension name for the auxiliary basis filename of each species

   :code:`pseudocharge`: define pseudocharge according to the adopted GTH pseudopotential
   
2. Initialize the systems used for the CP2K calculation by running:

   :code:`python3 -m salted.cp2k.xyz2sys`

   System cells and coordinates will be automatically saved in folders named :code:`conf_1`, :code:`conf_2`, ... up to the total number of structures included in the xyz dataset file, located into the selected :code:`inp.qm.path2qm`. In the dataset file, the cell must be put before each configuration coordinates even if it does not change.
   
2. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`. It gives a single output file for all the species, of the contracted basis. The file may be corrupted if the printed numbers have too many digits. In this case, a modification to the cp2k source must be direclty done. From the output file, create the auxiliary basis file per atom by running:

   :code:`python3 -m salted.cp2k.extract_basis cp2k_basis_filename`

with cp2k_basis_filename the name of the output file.

3. Add the selected auxiliary basis to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

   This creates a basis directory in the salted_path given in the input file, which will be used.

4. Run the CP2K calculations for each configuration in the created directories at step 2, using the selected auxiliary basis and print out the training data made of reference RI coefficients and 2-center auxiliary integrals. An example of a CP2K input file can be found in :code:`cp2k-inputs/RI-print.inp`.

5. Set the :code:`inp.qm.coeffile` and :code:`inp.qm.ovlpfile` variables according to the filename of the generated training data and convert them to SALTED format by running:

   :code:`python3 -m salted.cp2k.cp2k2salted`
