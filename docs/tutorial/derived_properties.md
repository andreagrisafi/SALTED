# Derived properties 

We describe here the calculation of derived properties depending on the chosen electronic-structure program.

## CP2K (`qmcode : cp2k`)

Analytical GTO-based calculations of derived electrostatic properties can be performed by relevant SALTED functions, as called from `salted.validation` or `salted.prediction` modules. We distinguish below whether the electron density or its electric-field response is considered as SALTED learning target.

1. `saltedtype : density`

    The function `compute_charge_and_dipole` is used to compute total electronic charges and dipole moments from input density coefficients. In this case, the total charge is first computed from the raw predicted coefficients; the L=0 (isotropic) components are then rescaled to enforce exact charge conservation, essential to compute total dipole moments without any origin dependence. **NB:** absolute values of dipole moments will only make physical sense when computed from electron densities that vanish before reaching the cell periodic boundaries.
    
    The function `compute_hartree_energy` is used to compute the total electrostatic energy of the system. As of now, this is only implemented by providing in input the 2-center Coulomb integral between auxiliary basis functions, according to the adopted density-fitting Coulomb metric, in addition to the density coefficients.
    
    When validating a SALTED model, `salted.validation` will automatically output the following files:
    
    - `charges.dat`: reference vs. predicted total electronic charge
     
    - `dipoles.dat`: reference vs. predicted total dipole moment of 3 elements each (X, Y, Z)
       
    - `electrostatic_energy.dat`: reference vs. predicted electrostatic energy (only for `dfmetric: coulomb`)

2. `saltedtype : density-response`

     The function `compute_polarizability` is used to compute polarizability tensors from input density-response coefficients. The total integral of the predicted density response is enforced to vanish by removing the total integral error from the L=0 coefficients for each Cartesian component. **NB:** absolute values of polarizabilities will only make physical sense when computed from electron-density responses that vanish before reaching the cell periodic boundaries.
     
     When validating a SALTED model, `salted.validation` will automatically output a `polarizabilities.dat` file including reference vs. predicted flattened rank-2 tensors of 9 elements each (XX, XY, XZ, YX, YY, ...). When predicting for a batch of structures via `salted.prediction`, an `alpha_only` keyword can be used in the `inp.prediction` section to only predict the L=0 and L=1 density-response coefficients, required for the calculation of the polarizability.

### Print 3D-fields as cube files (optional)

Electron densities, total charge densities, electrostatic potentials and electric fields associated with density-fitted or SALTED-predicted coefficients can be printed on a 3D real-space grid as `<cube_file_name>.cube` files, via the function `salted.cp2k.cube_reconstruction` (MPI parallelizable). The script `print_cubes.py` found in the example folder provides a minimal working example. A light 3D grid is used by default for visualization purposes; alternatively, reference cube files can be provided in input to use a prescribed 3D grid, as well as to measure the mean absolute error of the predicted electron density, normalized by the total number of electrons (% MAE). In principle, similar electron-density cube files can directly be provided in the CP2K input as initial guess for the SCF cycle.

## FHI-aims (`qmcode : aims`)

A description of how to restart a DFT calculation using predicted SALTED coefficients to access derived DFT properties is reported in [the dedicated SALTED/FHI-aims tutorial](https://fhi-aims-club.gitlab.io/tutorials/fhi-aims-with-salted).

## PySCF (`qmcode : pyscf`)

It is possible to validate the model by computing the total electrostatic energy and compare it against the reference PySCF values.

1. Calculate the reference electrostatic energies by running:
    ```bash
    python -m salted.pyscf.electro_energy
    ```
2. Calculate the energies derived from the predicted densities on the validation set and evaluate the error in kcal/mol, by running:
    ```bash
    python -m salted.pyscf.electro_error
    ```
