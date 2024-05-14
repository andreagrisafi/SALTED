# Prepare Dataset

This section describes how to prepare the dataset for training the SALTED model with different ab initio software packages.

## What do we need?

1. Product basis overlap matrices
1. Density fitting coefficients

## Generate Dataset

To date, support for generating these overlap matrices and coefficients is included in three electronic structure packages - PySCF, FHI-aims and CP2K. If you develop another package and would like to develop SALTED integration, please contact one of the developers.

Whichever code is used, the result should be the generation of new directories named `overlaps` and `coefficients` in the `saltedpath` directory. These will be used to train a SALTED model as described in the next section.

### PySCF

1. The following input arguments must be added to the `inp.qm` section:  
    - `qmcode`: define the quantum-mechanical code as `pyscf`
    - `path2qm`: set the path where the PySCF data are going to be saved 
    - `qmbasis`: define the wave function basis set for the Kohn-Sham calculation (example: `cc-pvqz`)
    - `functional`: define the functional for the Kohn-Sham calculation (example: `b3lyp`)
1. Define the auxiliary basis set using the input variable `dfbasis`, as provided in the `inp.qm` section. This must be chosen consistently with the wave function basis set (example: `RI-cc-pvqz`). Then, add this basis set information to SALTED by running:
```bash
   python3 -m salted.get_basis_info
```
1. Run PySCF to compute the Kohn-Sham density matrices: 
```bash
   python3 -m salted.pyscf.run_pyscf
```
1. From the computed density matrices, perform the density fitting on the selected auxiliary basis set by running: 
```bash
   python3 -m salted.pyscf.dm2df
```

### FHI-aims

A detailed description of how to generate the training data for SALTED using FHI-aims can be found at [the dedicated SALTED/FHI-aims tutorial](https://gitlab.com/FHI-aims-club/tutorials/fhi-aims-with-salted/-/blob/optimization/Tutorial/Tutorial-2/README.md?ref_type=heads&plain=1>).


### CP2K

1. The following input arguments must be added to the `inp.qm` section:
    - `qmcode`: define quantum-mechanical code as `cp2k`
    - `path2qm`: set the path where the CP2K data are going to be saved
    - `periodic`: set the periodicity of the system (`0D,2D,3D`)
    - `coeffile`: filename of RI density coefficients as printed by CP2K
    - `ovlpfile`: filename of 2-center auxiliary integrals as printed by CP2K
    - `dfbasis`: define auxiliary basis for the electron density expansion
    - `pseudocharge`: define pseudocharge according to the adopted GTH pseudopotential
1. Initialize the systems used for the CP2K calculation by running:
```bash
   python3 -m salted.cp2k.xyz2sys
```
   System cells and coordinates will be automatically saved in folders named `conf_1`, `conf_2`, ... up to the total number of structures included in the dataset, located into the selected `inp.qm.path2qm`. 
1. Print auxiliary basis set information from the CP2K automatically generated RI basis set, as described in https://doi.org/10.1021/acs.jctc.6b01041. An example of a CP2K input file can be found in `cp2k-inputs/get_RI-AUTO_basis.inp`. 
1. An uncontracted version of this basis can be produced to increase the efficiency of the RI printing workflow, by running:
```bash
   python3 -m salted.cp2k.uncontract_ri_basis contracted_basis_file uncontracted_basis_file
```
   Then, copy `uncontracted_basis_file` to the `cp2k/data/` folder in order to use this basis set to produce the reference density-fitting data, and set the corresponding filename in the `inp.qm.dfbasis` input variable.
1. Add the selected auxiliary basis to SALTED by running:
```bash
   python3 -m salted.get_basis_info
```
1. Run the CP2K calculations using the selected auxiliary basis and print out the training data made of reference RI coefficients and 2-center auxiliary integrals. An example of a CP2K input file can be found in `cp2k-inputs/qmmm_RI-print.inp`. 
1. Set the `inp.qm.coeffile` and `inp.qm.ovlpfile` variables according to the filename of the generated training data and convert them to SALTED format by running:
```bash
   python3 -m salted.cp2k.cp2k2salted
```
