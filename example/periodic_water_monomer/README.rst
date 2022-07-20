#-------------------------------------------------------------------------------
# SETUP
#-------------------------------------------------------------------------------

Before beginning, run
:code:`source YOUR_SALTED_DIRECTORY/env.sh`
and 
:code:`source YOUR_TENSOAP_DIRECTORY/env.sh`

Ensure that the file $SALTEDPATH/basis.py contains an entry corresponding to the dfbasis you wish to use

#-------------------------------------------------------------------------------
# GENERATE TRAINING DATA USING AIMS
#-------------------------------------------------------------------------------

Generate the overlap matrices and projections for the configurations of water using FHI-AIMS.

First, generate AIMS geometry input files from the xyz file, running
:code:`python make_geoms.py`

To run AIMS for each configuration, run
:code:`./run-aims.sh`
or
:code:`sbatch run-aims.sbatch`.
In either case, $QMDIR must be set to the same directory as path2qm in inp.py and the path to the AIMS executable must be specified. Further changes to the submission script may be required.

The AIMS output must be re-ordered and the Condon-Shottley convention applied before being input to SALTED. To do this, run
:code:`mpirun -np $ntasks python move_data.py`
This also removes the overlap matrix and projections vector from the AIMS output folder to save space.

IMPORTANT NOTE: The auxiliary basis used by AIMS depends sensitively on a number of choices made in control.in. Please check that the information about the auxiliary basis in `$SALTEDPATH/basis.py` matches that output by FHI-aims in `basis_info.out`. To facilitate integration between AIMS and SALTED, the following script generates a dictionary entry which can be appended to `$SALTEDPATH/basis.py` containing the necessary information about the auxiliary basis used to generate the training data:
:code:`python get_basis_info.py`

Calculate the spherically averaged baseline coefficients across the training set
:code:`python $SALTEDPATH/get_averages.py`

#-------------------------------------------------------------------------------
# GENERATE DESCRIPTORS
#-------------------------------------------------------------------------------

Calculate the lambda-SOAP descriptors, using
:code:`python $SALTEDPATH/run-tensoap.py`

The number of sparse features and number of structures used for sparsification can be specified using the flags `-nc` and `-ns` respectively. The flag `-p` should be added to handle periodic systems.

#-------------------------------------------------------------------------------
# PERFORM SALTED MINIMISATION AND VALIDATION
#-------------------------------------------------------------------------------

Compute descriptors per basis function type for a given training set
:code:`python $SALTEDPATH/rkhs.py`

Compute global feature vector and save as sparse object 
:code:`mpirun -np $ntasks python $SALTEDPATH/feature_vector.py`

Minimize loss function and print out regression weights
:code:`mpirun -n $ntasks python $SALTEDPATH/minimize_loss.py`

Validate model predicting on the remaining structures
:code:`mpirun -np $ntasks python $SALTEDPATH/validation.py`

To evaluate the properties derived from these validation predicted densities, run
:code:`sbatch run-aims-validate.sbatch`.
The energies can be collected from the raw output files into numpy arrays using the script
:code:`python collect_energies.py`


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
