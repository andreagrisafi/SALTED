#---------------------------------------------------
# GENERATE TRAINING DATA
#---------------------------------------------------

The following input arguments must be added to the :code:`inp.qm` section:
    
:code:`qmbasis`: define the wave function basis set for the Kohn-Sham calculation (example: :code:`cc-pvqz`)
:code:`functional`: define the functional for the Kohn-Sham calculation (example: :code:`b3lyp`)

Run PySCF to compute the Kohn-Sham density matrices: 

:code:`python3 -m salted.pyscf.run_pyscf`

Define the auxiliary basis set using the input variable :code:`dfbasis`, as provided in the :code:`inp.qm` section. This must be chosen consistently with the wave function basis set (example: :code:`RI-cc-pvqz`). Then, add this basis set information to SALTED by running:

:code:`python3 -m salted.get_basis_info`

From the computed density matrices, perform the density fitting on the selected auxiliary basis set by running: 

:code:`python3 -m salted.pyscf.dm2df`

#---------------------------------------------------
# INDIRECT PREDICTION OF ELECTROSTATIC ENERGY
#---------------------------------------------------

Calculate the reference energies of the water molecules used in validation, using

:code:`python3 salted.pyscf.electro_energy`

Calculate the energies derived from the predicted densities on the validation set and evaluate the error in kcal/mol, by running:

:code:`python3 salted.pyscf.electro_error`
