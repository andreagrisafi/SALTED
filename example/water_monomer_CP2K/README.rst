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

6. Run the density fitting script for the required configurations:

   :code:`python3 -m salted.cp2k.density_fitting conf_start conf_end` (MPI parallelizible)

The resulting fitting coefficients and overlap matrices are saved in the :code:`coefficients` and :code:`overlaps` folders of :code:`inp.salted.saltedpath`.

7. Validate the trained model:

   :code:`python3 -m salted.validation` (MPI parallelizable)

   The validation script computes:

   - :code:`errors.dat`: RMSE of the predicted density
   - :code:`charges.dat`: reference vs. predicted total electronic charge
   - :code:`dipoles.dat`: reference vs. predicted total dipole moment
   - :code:`electrostatic_energy.dat`: reference vs. predicted Hartree energy (only for `dfmetric: coulomb`)

   The total charge is computed **first** from the raw predicted coefficients, so that `charges.dat` reports the actual charge error of the model.
   The predicted isotropic components are then rescaled to absorb that error and conserve the charge exactly, and these charge-corrected coefficients are used for the dipole moment and the Hartree energy.

8. Test density reconstruction (optional)

   The fitted coefficients can be expanded back onto a real-space grid and written as Gaussian cube files if needed, via :code:`salted.cp2k.cube_reconstruction`.
   The script :code:`test_cube.py` provides a minimal working example.