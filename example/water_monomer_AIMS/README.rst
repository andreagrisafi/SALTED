#-------------------------------------------------------------------------------
# GENERATE TRAINING DATA USING AIMS
#-------------------------------------------------------------------------------

Generate the overlap matrices and projections for the configurations of water using FHI-AIMS.

First, generate AIMS geometry input files from the xyz file, running
:code:`python -m salted.aims.make_geoms`

To run AIMS and generate the training data for each configuration, run
:code:`sbatch run-aims.sbatch`.
In the submission script, $QMDIR must be set to the same directory as path2qm in inp.py, and the path to the AIMS executable must be specified. Further changes to the submission script may be required depending on your HPC setup.

The AIMS output must be re-ordered and the Condon-Shottley convention applied before being input to SALTED. To do this, run
:code:`mpirun -np $ntasks python -m salted.aims.move_data`
This removes the overlap matrix and projections vector from the AIMS output folder to save space, and creates re-ordered numpy files in the folders "projections", "overlaps" and "coefficients".

IMPORTANT NOTE: The auxiliary basis used by AIMS depends sensitively on a number of choices made in control.in. Please check that the information about the auxiliary basis in `$SALTEDPATH/basis.py` matches that output by FHI-aims in `basis_info.out`. To facilitate integration between AIMS and SALTED, the following script generates a dictionary entry called `new_basis_entry` which will be automaticallyappended to SALTED's internal list of bases, which contains the necessary information about the auxiliary basis used to generate the training data:
:code:`python -m salted.aims.get_basis_info`

To check the accuracy of the auxilliary basis, run :code:`python -m salted.aims.get_df_err`. This will produce a file called df_maes listing the percentage integrated mean absolute error in the density for every structure in the dataset. In this example it should be just over 0.1% for each structure.

Calculate the spherically averaged baseline coefficients across the training set
:code:`python -m salted.get_averages`

#-------------------------------------------------------------------------------
# GENERATE DESCRIPTORS
#-------------------------------------------------------------------------------

(The submission script `run-ml.sbatch` is provided for convenience to run the following steps)

Calculate the lambda-SOAP descriptors, using
:code:`mpirun -np $ntasks python -m salted.equirepr`

The number of sparse features and number of structures used for sparsification can be specified using the flags `inp.ncut` and `inp.nsamples` respectively. Setting `inp.ncut=-1` will not use any sparsification, which is recommended for this example.

These descriptors are then be further sparsified by selecting a representative number of reference environments, defined by `inp.Menv`, using
:code:`python -m salted.sparsify`
Note that this code should NOT be run using MPI parallelisation, even when it is being used for other functions.

#-------------------------------------------------------------------------------
# PERFORM SALTED MINIMISATION AND VALIDATION
#-------------------------------------------------------------------------------

Compute descriptors per basis function type for a given training set
:code:`mpirun -np $ntasks python -m salted.rkhs`

Compute global feature vector and save as sparse object 
:code:`mpirun -np $ntasks python -m salted.feature_vector`

There are two methods to calculate the regression weights. 

EITHER:

For smaller problems, they can be found via a matrix inversion (see Ref 3. in the main README). This is carried out via a two- or three- step process. For very small problems, the matrix can be built in a single calculation by running:
:code: `python -m salted.matrices`.
This will be the case if `inp.blocksize` is not present, or is not a positive integer. For this usage the code should be run serially. 

Alternatively, the matrix can be constructed in several blocks, each constructed from an equally sized subset of the training set. This will be the case if `inp.blocksize` is a positive integer. Note that `inp.blocksize` must be an exact divisor of the training set size determined by `inp.Ntrain*inp.trainfrac`. If calculating the matrix blockwise, the code can be run in parallel:
:code: `mpirun -np $ntasks python -m salted.matrices`.
In either case, the training set will either be chosen as a `sequential` or `random` selection, depending on the value of `inp.trainsel`.

The blocks then need to be combined to form a single matrix; this is done by running (serially):
:code: `python -m salted.collect_matrices`.

Finally, the matrix is inverted:
:code: `python -m salted.regression`.

OR:

For large problems, the regrssion weights should be found by minimizing the loss function directly. This requires just a single step:
:code:`mpirun -n $ntasks python -m salted.minimize_loss`

After finding the regression weights by either method, validate the model by predicting the density of the remaining structures
:code:`mpirun -np $ntasks python -m salted.validation`
This produces the predicted coefficients for each configuration in the validation set. These are output to validations_+inp.saltedname. The average error for this example should be 0.75%.

#-------------------------------------------------------------------------------
# PREDICT DENSITIES OF NEW STRUCTURES
#-------------------------------------------------------------------------------

The resulting model can be used to predict the expansion coefficients of further structures not used in training the model. These structures will be read from `inp.filename`; this must be changed from the file used to train the model. Calculate the predicted coefficients using
:code:`mpirun $ntasks python -m salted.equipred`

The predicted coefficients are output to "predictions_"+inp.saltedname+inp.predname.

#-------------------------------------------------------------------------------
# CALCULATING DERIVED PROPERTIES OF NEW STRUCTURES
#-------------------------------------------------------------------------------

To evaluate the accuracy of the predicted densities and their derived properties, run
:code:`sbatch run-aims-predict.sbatch`.
Once again, in the submission script $DATADIR must be set to the same directory as path2qm+predict_data in inp.py, and the path to the AIMS executable must be specified. Further changes to the submission script may be required depending on your HPC setup.

To calculate the reference result, run
:code:`sbatch run-aims.sbatch`
after modifying $DATADIR to be set to the inp.path2qm+inp.predict_data. In this case, the keywords prefixed by "ri_" may be commented from control.in, apart from ri_output_density_only.

The accuracy of the densities can be calculated by running :code:`python -m salted.aims.get_ml_err`. This will produce a file called ml_maes listing the percentage integrated mean absolute error in the density for every structure in the dataset.

The electrostatic, XC and total energies per atom can be collected from the raw output files into numpy arrays using the script
:code:`python -m salted.aims.collect_energies`
The reference energies are output to files with the prefix `predict_reference`, and the energied derived from the densities produced by SALTED to files with the prefix `predict`. Other properties can be collected with simple modifications to the script, if desired.

