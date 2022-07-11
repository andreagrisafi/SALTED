#-------------------------------------------------------------------------------
# SETUP
#-------------------------------------------------------------------------------

Before beginning, run
:code:`source YOUR_SALTED_DIRECTORY/env.sh`
and 
:code:`source YOUR_TENSOAP_DIRECTORY/env.sh`

Ensure that the file $SALTEDPATH/basis.py contains an entry corresponding to the dfbasis you wish to use

#-------------------------------------------------------------------------------
# GENERATE TRAINING DATA
#-------------------------------------------------------------------------------

Generate the density matrices for the configurations of water using PySCF using
:code:`python $SALTEDPATH/run-pyscf.py`

Calculate the projections of these density matrices onto auxiliary basis functions using
:code:`python $SALTEDPATH/dm2df-pyscf.py`

Each of these commands can be run with the flag `-iconf n` to generate the training data for the nth structure only.

Calculate the spherically averaged baseline coefficients across the training set
:code:`python $SALTEDPATH/get_averages.py`

#-------------------------------------------------------------------------------
# GENERATE DESCRIPTORS
#-------------------------------------------------------------------------------

Calculate the lambda-SOAP descriptors, using
:code:`python $SALTEDPATH/run-tensoap.py`

The number of sparse features and number of structures used for sparsification can be specified using the flags `-nc` and `-ns` respectively

#-------------------------------------------------------------------------------
# PERFORM SALTED MINIMISATION AND VALIDATION
#-------------------------------------------------------------------------------

Compute descriptors per basis function type for a given training set
:code:`python $SALTEDPATH/rkhs.py`

Compute global feature vector and save as sparse object 
:code:`mpirun -np $ntasks python $SALTEDPATH/feature_vector.py`

Minimize loss function and print out regression weights
:code:`mpirun -n $ntasks python $SALTEDPATH/minimize_loss-parallel.py`
or if it is necessary to run serially
:code:`python .$SALTEDPATH/minimize_loss-serial.py`

Validate model predicting on the remaining structures
:code:`mpirun -np $ntasks python $SALTEDPATH/validation.py` 

#-------------------------------------------------------------------------------
# PREDICT DENSITIES OF NEW STRUCTURES
#-------------------------------------------------------------------------------

Calculate the lambda-SOAP descriptors for the structures to predict, using
:code:`python $SALTEDPATH/run-tensoap-predict.py`

Compute descriptors per basis function type for the prediction set
:code:`python $SALTEDPATH/rkhs-prediction.py`

Calculate the predicted coefficients using
:code:`python $SALTEDPATH/prediction.py`

This produces the predicted coefficients for each configuration found in inp.predict_filename. These are output to inp.path2qm+inp.preddir.

#-------------------------------------------------------------------------------
# INDIRECT PREDICTION OF ELECTROSTATIC ENERGY
#-------------------------------------------------------------------------------

Calculate the reference energies of the water molecules used in validation, using
:code:`python $SALTEDPATH/electro_energy-pyscf.py`

Calculate the energies derived from the predicted densities and evaluate the error, using
:code:`python $SALTEDPATH/electro_error-pyscf.py`

To compare performance to an equivalent 'direct' prediction, run
:code:`python $SALTEDPATH/sparse-gpr_energies.py`


#-------------------------------------------------------------------------------
# PERFORM ORTHOGONALISED MINIMISATION AND VALIDATION
#-------------------------------------------------------------------------------

:code:`python $SALTEDPATH/ortho_projections.py`
:code:`python $SALTEDPATH/ortho_regression.py`
:code:`python $SALTEDPATH/ortho_error.py`
