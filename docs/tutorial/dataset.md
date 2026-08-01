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

A detailed description of how to generate the training data for SALTED using FHI-aims can be found at [the dedicated SALTED/FHI-aims tutorial](https://fhi-aims-club.gitlab.io/tutorials/fhi-aims-with-salted).


### CP2K

1. The following input arguments must be included in the `inp.qm` section:
    - `qmcode`: define quantum-mechanical code as `cp2k`
    - `path2qm`: set the path where the CP2K data are going to be saved
    - `periodic`: set the periodicity of the system (`0D,2D,3D`)
    - `dfbasis`: RI (density-fitting) basis filename appended for each species, extracted from CP2K
    - `dfmetric`: metric used to perform the density fitting (`identity` or `coulomb`)
2. Initialize the systems used for the CP2K calculation:
    ```bash
    python3 -m salted.cp2k.xyz2sys
    ```
   System cells and coordinates are extracted from the configuration dataset in XYZ format and saved in folders named `conf_1`, `conf_2`, ... located in the path `inp.qm.path2qm`. **NB:** cell information (`Lattice`) must be included in the second line of each XYZ configuration, even if it does not change.
3. Run SCF calculations and save the electron density cube file for each configuration in the corresponding folders previously generated. An example CP2K input is provided in `cp2k-inputs/SCF.inp`. The density must be printed through the `E_DENSITY_CUBE` section, so that each `conf_N` folder contains a file matching `*ELECTRON_DENSITY-1_0.cube`. **NB:** The cube grid is assumed to be orthorhombic.
4. Print the RI basis set information required for SALTED postprocessing of the CP2K density. An example CP2K input is provided in :code:`cp2k-inputs/RI-basis.inp`. This operation only needs to be performed once for any arbitrary configuration included in the dataset adopting the given choice of RI basis. The output is a single file including wavefunction and RI basis set information of all the species included in the selected test configuration. The information about the pseudopotential used is extracted from a CP2K input file, and the potential file used by CP2K, by running:
   ```bash
   python3 -m salted.cp2k.extract_basis_and_pseudopotential cp2k_basis_filename cp2k_input_filename pseudopotential_filename
   ```
   with `cp2k_basis_filename` the output basis set filename. This will create a separate file for each species in the format, e.g., H-:code:`dfbasis`, O-:code:`dfbasis` for the basis, and H-`local_pseudo.dat`, O-`local_pseudo.dat` for the pseudo-charge and pseudopotential radius. Warning: CP2K v2026.2 must be used here.
5. Add the RI basis set information to SALTED:
    ```bash
    python3 -m salted.get_basis_info
    ```
6. Perform the density fitting for the selected range of configurations:
    ```bash
    python3 -m salted.cp2k.density_fitting conf_start conf_end
    ```