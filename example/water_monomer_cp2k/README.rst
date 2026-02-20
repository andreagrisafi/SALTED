Generate training data using CP2K
----------------------------------

In what follows, we describe how to generate training electron densities of a dataset made of water monomers and dimers, using the CP2K simulation program. NB: this is made possible through to the official development version of CP2K (https://github.com/cp2k/cp2k).

1. The file inp.yaml is the input file used by salted routines (the description of all keywords is available at https://salted.readthedocs.io/en/latest/input/). The following input arguments must be added to the :code:`inp.qm` section in it:

   :code:`qmcode`: define quantum-mechanical code as :code:`cp2k`

   :code:`path2qm`: set the path where the CP2K data are going to be saved

   :code:`periodic`: set the periodicity of the system (:code:`0D,2D,3D`)

   :code:`coeffile`: filename of RI density coefficients as printed by CP2K

   :code:`ovlpfile`: filename of 2-center auxiliary integrals as printed by CP2K

   :code:`dfbasis`: extension name for the auxiliary basis filename of each species

   :code:`pseudocharge`: define pseudocharge according to the adopted GTH pseudopotential
   
2. Initialize the systems used for the CP2K calculation by running:

   :code:`python3 -m salted.cp2k.xyz2sys`

   System cells and coordinates are extracted from the dataset of configurations and are saved in folders named :code:`conf_1`, :code:`conf_2` ...  in the path :code:`inp.qm.path2qm`. It is required that in the dataset file, the cell must be written before each configuration coordinates, even if it does not change.

3. Run a converged SCF CP2K calculation and save the obtained wavefunction for each configuration, using the input file :code:`cp2k-inputs/SCF-print.inp`. For this, the input file can be copy in each folders created at the step before and run from there. The writing of a script to automatically do these steps is useful.

4. Restart the CP2K calculation for each configuration, to print the RI fitting coefficients and the overlap integrals using `cp2k-inputs/RI-print.inp` . The auxiliary basis used to project the density is specified in this input.
   
5. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`. This can be done using any configuration, as the auxiliary basis used is the same for all configurations. The output gives a single file for all the species, of the contracted basis of the auxiliary one. Run

   :code:`python3 -m salted.cp2k.extract_basis cp2k_basis_filename`

with cp2k_basis_filename the name of the output file to create one file per atom.

6. Add the selected auxiliary basis to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

7. Set the :code:`inp.qm.coeffile` and :code:`inp.qm.ovlpfile` variables according to the filename of the generated training data and convert them to SALTED format by running:

   :code:`python3 -m salted.cp2k.cp2k2salted`
