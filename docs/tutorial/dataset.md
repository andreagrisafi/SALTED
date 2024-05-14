# Prepare Dataset

This section describes how to prepare the dataset for training the SALTED model with different ab initio software packages.

## What we need?

1. Product basis overlap matrices
1. Density fitting coefficients

## Generate Dataset

To date, support for generating these overlap matrices and coefficients is included in three electronic structure packages - PySCF, FHI-aims and CP2K. If you develop another package and would like to develop SALTED integration, please contact one of the developers.

Whichever code is used, the result should be the generation of new directories named `overlaps` and `coefficients` in the `saltedpath` directory. These will be used to train a SALTED model as described in the next section.

### PySCF

1. The following input arguments must be added to the :code:`inp.qm` section:

   :code:`qmcode:`: define the quantum-mechanical code as :code:`pyscf`

   :code:`path2qm`: set the path where the PySCF data are going to be saved
    
   :code:`qmbasis`: define the wave function basis set for the Kohn-Sham calculation (example: :code:`cc-pvqz`)

   :code:`functional`: define the functional for the Kohn-Sham calculation (example: :code:`b3lyp`)

2. Define the auxiliary basis set using the input variable :code:`dfbasis`, as provided in the :code:`inp.qm` section. This must be chosen consistently with the wave function basis set (example: :code:`RI-cc-pvqz`). Then, add this basis set information to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

3. Run PySCF to compute the Kohn-Sham density matrices: 

   :code:`python3 -m salted.pyscf.run_pyscf`

4. From the computed density matrices, perform the density fitting on the selected auxiliary basis set by running: 

   :code:`python3 -m salted.pyscf.dm2df`

### FHI-aims

A detailed description of how to generate the training data for SALTED using FHI-aims can be found at [the dedicated SALTED/FHI-aims tutorial](https://gitlab.com/FHI-aims-club/tutorials/fhi-aims-with-salted/-/blob/optimization/Tutorial/Tutorial-2/README.md?ref_type=heads&plain=1>).


### CP2K

1. The following input arguments must be added to the :code:`inp.qm` section:

   :code:`qmcode`: define quantum-mechanical code as :code:`cp2k`

   :code:`path2qm`: set the path where the CP2K data are going to be saved

   :code:`periodic`: set the periodicity of the system (:code:`0D,2D,3D`)

   :code:`coeffile`: filename of RI density coefficients as printed by CP2K

   :code:`ovlpfile`: filename of 2-center auxiliary integrals as printed by CP2K

   :code:`dfbasis`: define auxiliary basis for the electron density expansion

   :code:`pseudocharge`: define pseudocharge according to the adopted GTH pseudopotential

2. Initialize the systems used for the CP2K calculation by running:

   :code:`python3 -m salted.cp2k.xyz2sys`

   System cells and coordinates will be automatically saved in folders named :code:`conf_1`, :code:`conf_2`, ... up to the total number of structures included in the dataset, located into the selected :code:`inp.qm.path2qm`. 

2. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file can be found in :code:`cp2k-inputs/get_RI-AUTO_basis.inp`. 

3. An uncontracted version of this basis can be produced to increase the efficiency of the RI printing workflow, by running:

   :code:`python3 -m salted.cp2k.uncontract_ri_basis contracted_basis_file uncontracted_basis_file`

   Then, copy :code:`uncontracted_basis_file` to the :code:`cp2k/data/` folder in order to use this basis set to produce the reference density-fitting data, and set the corresponding filename in the :code:`inp.qm.dfbasis` input variable.

4. Add the selected auxiliary basis to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

5. Run the CP2K calculations using the selected auxiliary basis and print out the training data made of reference RI coefficients and 2-center auxiliary integrals. An example of a CP2K input file can be found in :code:`cp2k-inputs/qmmm_RI-print.inp`. 

6. Set the :code:`inp.qm.coeffile` and :code:`inp.qm.ovlpfile` variables according to the filename of the generated training data and convert them to SALTED format by running:

   :code:`python3 -m salted.cp2k.cp2k2salted` 

