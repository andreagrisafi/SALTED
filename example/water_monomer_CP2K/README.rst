Generate training data using CP2K (from v2026.2)
------------------------------------------------

In what follows, we describe how to generate training electron densities to be used in SALTED using the CP2K electronic-structure program. 

1. The following input arguments must be included in the :code:`inp.qm` section:

   :code:`qmcode`: define quantum-mechanical code as :code:`cp2k`

   :code:`path2qm`: set the path where the CP2K data are going to be saved

   :code:`periodic`: set the periodicity of the system (:code:`0D,2D,3D`)

   :code:`dfbasis`: RI (density-fitting) basis filename appended for each species, extracted from CP2K 

   :code:`dfmetric`: metric used to perform the density fitting (:code:`identity` or :code:`coulomb`)
   
2. Initialize the systems used for the CP2K calculation by running:

   :code:`python3 -m salted.cp2k.xyz2sys`

   System cells and coordinates are extracted from the configuration dataset in XYZ format and saved in folders named :code:`conf_1`, :code:`conf_2`, ...  located in the path :code:`inp.qm.path2qm`. NB: cell information (Lattice) must be included in second line of each XYZ configuration, even if it does not change.

3. Run SCF calculations for each configuration in the corresponding folders previously generated, printing the converged electron density on the real-space grid. An example CP2K input is provided in :code:`cp2k-inputs/SCF.inp`.

4. Print the RI basis set information required for SALTED postprocessing of the CP2K density. An example CP2K input is provided in :code:`cp2k-inputs/RI-basis.inp`. This operation can be performed only once for any arbitrary configuration included in the dataset adopting the given choice of RI basis. The output is a single file including wavefunction and RI basis set information of all the species included in the selected test configuration. The information about the pseudopotential used is extracted from a CP2K input file, and the potential file used by CP2K. To extract this information run:

   :code:`python3 -m salted.cp2k.extract_basis_and_pseudopotential cp2k_basis_filename cp2k_input_filename pseudopotential_filename`

   with :code:`cp2k_basis_filename` the output basis set filename. This will create a separate file for each species in the format, e.g., H-:code:`dfbasis`, O-:code:`dfbasis` for the basis, and H-:code:`local_pseudo.dat`, O-:code:`local_pseudo.dat` for the pseudo-charge and local pseudopotential radius.

5. Add the RI basis set information to SALTED by running:

   :code:`python3 -m salted.get_basis_info`

6. Perform the density fitting (either with :code:`identity` or :code:`coulomb` metric) on the selected RI basis for the required configurations:

   :code:`python3 -m salted.cp2k.density_fitting conf_start conf_end` (MPI parallelizible)

   A Lagrange multiplier is adopted to solve the linear problem under total charge conservation. The fitted coefficients and 2-center integral (overlap) matrices are saved in the :code:`coefficients` and :code:`overlaps` folders, respectively, in :code:`inp.salted.saltedpath`.

Derived properties
------------------

Analytical calculation of derived electrostatic properties is performed by relevant SALTED functions, e.g., :code:`salted.validation`, :code:`salted.prediction` and :code:`salted.salted_prediction`. 

1. :code:`saltedtype : density`:

   The total charge is first computed from the raw predicted coefficients. The L=0 (isotropic) components are then rescaled to enforce exact charge conservation and compute total dipole moments and Hartree energies. Specifically, :code:`salted.validation` will automatically output the following files:
   
      - :code:`charges.dat`: reference vs. predicted total electronic charge
      - :code:`dipoles.dat`: reference vs. predicted total dipole moment of 3 elements each (X, Y, Z)
      - :code:`electrostatic_energy.dat`: reference vs. predicted electrostatic energy (only for `dfmetric: coulomb`)

2. :code:`saltedtype : density-response`:

   The total integral of the predicted density response is enforced to vanish by removing the total integral error from the L=0 coefficients for each Cartesian component. Derived polarizability tensors are then analytically computed. :code:`salted.validation` will automatically output a :code:`polarizabilities.dat` file including reference vs. predicted flattened rank-2 tensors of 9 elements each (XX, XY, XZ, YX, YY, ...). NB: a :code:`alpha_only` keyword can be used in the :code:`inp.prediction` section to only predict the L=0 and L=1 density-response coefficients, required for the calculation of the polarizability. 


Print 3D-fields as cube files (optional)
----------------------------------------

Electron densities, total charge densities, electrostatic potentials and electric fields associated with density-fitted or SALTED-predicted coefficients can be printed on a 3D real-space grid as :code:`<cube_file_name>.cube` files, via the function :code:`salted.cp2k.cube_reconstruction`. The script :code:`print_cubes.py` found in the example folder provides a minimal working example. A light 3D grid is used by default for visualization purposes; alternatively, reference cube files can be provided in input to use a prescribed 3D grid, as well as to measure the mean absolute error of the electron density, normalized by the total number of electrons (% MAE).  
