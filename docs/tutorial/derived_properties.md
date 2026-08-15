# Derived properties with CP2K

Analytical calculation of derived electrostatic properties is performed by relevant SALTED functions, e.g., `salted.validation`, `salted.prediction` and `salted.salted_prediction`.

1. `saltedtype : density`

   The total charge is first computed from the raw predicted coefficients. The L=0 (isotropic) components are then rescaled to enforce exact charge conservation and compute total dipole moments and Hartree energies. Specifically, `salted.validation` will automatically output the following files:
    - `charges.dat`: reference vs. predicted total electronic charge
    - `dipoles.dat`: reference vs. predicted total dipole moment of 3 elements each (X, Y, Z)
    - `electrostatic_energy.dat`: reference vs. predicted electrostatic energy (only for `dfmetric: coulomb`)

2. `saltedtype : density-response`

   The total integral of the predicted density response is enforced to vanish by removing the total integral error from the L=0 coefficients for each Cartesian component. Derived polarizability tensors are then analytically computed. `salted.validation` will automatically output a `polarizabilities.dat` file including reference vs. predicted flattened rank-2 tensors of 9 elements each (XX, XY, XZ, YX, YY, ...). **NB:** an `alpha_only` keyword can be used in the `inp.prediction` section to only predict the L=0 and L=1 density-response coefficients, required for the calculation of the polarizability.

## Print 3D-fields as cube files (optional)

Electron densities, total charge densities, electrostatic potentials and electric fields associated with density-fitted or SALTED-predicted coefficients can be printed on a 3D real-space grid as `<cube_file_name>.cube` files, via the function `salted.cp2k.cube_reconstruction` (MPI parallelizable). The script `print_cubes.py` found in the example folder provides a minimal working example. A light 3D grid is used by default for visualization purposes; alternatively, reference cube files can be provided in input to use a prescribed 3D grid, as well as to measure the mean absolute error of the electron density, normalized by the total number of electrons (% MAE).
