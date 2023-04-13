#-------------------------------------------------------------------------------
# SETUP
#-------------------------------------------------------------------------------

Before beginning, run
:code:`source YOUR_SALTED_DIRECTORY/env.sh`
and 
:code:`source YOUR_TENSOAP_DIRECTORY/env.sh`

Ensure that the file $SALTEDPATH/basis.py contains an entry corresponding to the dfbasis you wish to use. Python scripts which can be run in parallel are indicated in this README; however, by setting inp.parallel=False, every SALTED script can be run serially.

#-------------------------------------------------------------------------------
# GENERATE TRAINING DATA USING AIMS
#-------------------------------------------------------------------------------

Generate the overlap matrices and projections for the configurations of water using FHI-AIMS.

First, generate AIMS geometry input files from the xyz file, running
:code:`python make_geoms.py`

To run AIMS and generate the training data for each configuration, run
:code:`sbatch run-aims.sbatch`.
In the submission script, $QMDIR must be set to the same directory as path2qm in inp.py, and the path to the AIMS executable must be specified. Further changes to the submission script may be required depending on your HPC setup.

The AIMS output must be re-ordered and the Condon-Shottley convention applied before being input to SALTED. To do this, run
:code:`mpirun -np $ntasks python move_data.py`
This also removes the overlap matrix and projections vector from the AIMS output folder to save space.

IMPORTANT NOTE: The auxiliary basis used by AIMS depends sensitively on a number of choices made in control.in. Please check that the information about the auxiliary basis in `$SALTEDPATH/basis.py` matches that output by FHI-aims in `basis_info.out`. To facilitate integration between AIMS and SALTED, the following script generates a dictionary entry called `new_basis_entry` which can be appended to `$SALTEDPATH/basis.py` containing the necessary information about the auxiliary basis used to generate the training data:
:code:`python get_basis_info.py`

To check the accuracy of the auxilliary basis, run :code:`python get_df_err.py`. This will produce a file called df_maes listing the percentage integrated mean absolute error in the density for every structure in the dataset. In this example it should be just over 0.1% for each structure.

Calculate the spherically averaged baseline coefficients across the training set
:code:`python $SALTEDPATH/get_averages.py`

#-------------------------------------------------------------------------------
# GENERATE DESCRIPTORS
#-------------------------------------------------------------------------------

(The submission script `run-ml.sbatch` is provided for convenience to run the following steps)

Calculate the lambda-SOAP descriptors, using
:code:`python $SALTEDPATH/run-tensoap.py -p -nc 0 --parallel $ntasks`

The number of sparse features and number of structures used for sparsification can be specified using the flags `-nc` and `-ns` respectively. These have respective default values of 1000 and 100. The flag `-p` should be added to handle periodic systems. Setting `-nc 0` will not use any sparsification, which is recommended for this example.

#-------------------------------------------------------------------------------
# PERFORM SALTED MINIMISATION AND VALIDATION
#-------------------------------------------------------------------------------

Compute descriptors per basis function type for a given training set
:code:`mpirun -np $ntasks python $SALTEDPATH/rkhs.py`

Compute global feature vector and save as sparse object 
:code:`mpirun -np $ntasks python $SALTEDPATH/feature_vector.py`

Minimize loss function and print out regression weights
:code:`mpirun -n $ntasks python $SALTEDPATH/minimize_loss.py`

Validate model predicting on the remaining structures
:code:`mpirun -np $ntasks python $SALTEDPATH/validation.py`
This produces the predicted coefficients for each configuration in the validation set. These are output to inp.path2qm+inp.valcdir. The average error for this example should be 0.75%.

#-------------------------------------------------------------------------------
# CALCULATING DERIVED PROPERTIES
#-------------------------------------------------------------------------------

To evaluate the properties derived from these validation predicted densities, run
:code:`sbatch run-aims-validate.sbatch`.
The electrostatic, XC and total energies per atom can be collected from the raw output files into numpy arrays using the script
:code:`python collect_energies.py`
The reference energies are output to files with the prefix `val_reference`, and the energied derived from the densities produced by SALTED to files with the prefix `validation`. Other properties can be collected with simple modifications to the script.

#-------------------------------------------------------------------------------
# PREDICT DENSITIES OF NEW STRUCTURES
#-------------------------------------------------------------------------------

Calculate the lambda-SOAP descriptors for the structures to predict, using
:code:`python $SALTEDPATH/run-tensoap.py -nc 0 --predict`
Note that we are predicting the density a cluster, despite having trained on periodic systems. AIMS treats periodic and non-periodic calculations on the same footing, allowing this.

Compute descriptors per basis function type for the prediction set
:code:`mpirun $ntasks python $SALTEDPATH/rkhs-prediction.py`

Calculate the predicted coefficients using
:code:`mpirun $ntasks python $SALTEDPATH/prediction.py`

This produces the predicted coefficients for each configuration found in inp.predict_filename. These are output to inp.path2qm+inp.predict_coefdir.

#-------------------------------------------------------------------------------
# CALCULATING DERIVED PROPERTIES OF NEW STRUCTURES
#-------------------------------------------------------------------------------

To evaluate the accuracy of the predicted densities and their derived properties, run
:code:`sbatch run-aims-predict.sbatch`.
Once again, in the submission script $DATADIR must be set to the same directory as path2qm+predict_data in inp.py, and the path to the AIMS executable must be specified. Further changes to the submission script may be required depending on your HPC setup.

The accuracy of the densities can be calculated by running :code:`python get_ml_err.py`. This will produce a file called ml_maes listing the percentage integrated mean absolute error in the density for every structure in the dataset. In this example it should be just over 0.6% on average.

The electrostatic, XC and total energies per atom can be collected from the raw output files into numpy arrays using the script
:code:`python collect_energies.py --predict`
The reference energies are output to files with the prefix `predict_reference`, and the energied derived from the densities produced by SALTED to files with the prefix `predict`. Other properties can be collected with simple modifications to the script.
