#---------------------------------------------------
# GENERATE TRAINING DATA
#---------------------------------------------------

Generate the density matrices for the configurations of water using PySCF using
:code:`python $SALTEDPATH/pyscf/run-pyscf.py`

Calculate the projections of these density matrices onto auxiliary basis functions using
:code:`python $SALTEDPATH/pyscf/dm2df-pyscf.py`

#---------------------------------------------------
# INDIRECT PREDICTION OF ELECTROSTATIC ENERGY
#---------------------------------------------------

Calculate the reference energies of the water molecules used in validation, using
:code:`python $SALTEDPATH/pyscf/electro_energy-pyscf.py`

Calculate the energies derived from the predicted densities and evaluate the error, using
:code:`python $SALTEDPATH/pyscf/electro_error-pyscf.py`
