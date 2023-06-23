#-------------------------------------------------------------------------------
# SETUP
#-------------------------------------------------------------------------------

Before beginning, run
:code:`source YOUR_SALTED_DIRECTORY/env.sh`
and 
:code:`source YOUR_TENSOAP_DIRECTORY/env.sh`

Ensure that the file $SALTEDPATH/basis.py contains an entry corresponding to the dfbasis you wish to use. Python scripts which can be run in parallel are indicated in this README; however, by setting inp.parallel=False, every SALTED script can be run serially.

#-------------------------------------------------------------------------------
# GENERATE TRAINING DATA USING CP2K
#-------------------------------------------------------------------------------

Define the path used to save the training quantum-mechanical data by setting the variable :code:`path2qm` in :code:`inp.py` and :code:`cd` there in order to generate the working directories for the reference CP2K calculations. A separate folder must be created for each structure in the dataset, 40 in this example, by running  
:code:`mkdir runs; cd runs; for i in {1..40}; do mkdir conf_$i ; done; cd ../`

Generate CP2K geometry input files from the xyz file, running
:code:`python xyz2sys.py`
NB: the script is specifically designed for 2D-periodic systems.

Copy the CP2K input file in the working directories:
:code:`for i in {1..40}; do cp gpw.inp YOUR_QM_PATH/runs/conf_$i/ ; done`

Run CP2K by modifying the :code:`cp2k.job` according to your HPC system.
:code:`sbatch cp2k.job`
At the end of each calculation, the density coefficients and overlap matrices are printed out in the folders previously generated. In this example, a RI_HFX SMALL basis set is automatically generated starting from a DZVP-MOLOPT-SR-GTH orbital basis in order to perform the density-fitting from the Kohn-Sham density matrix. 

#------------------------------------------------------------------------------
# GENERATION OF AUXILIARY BASIS
#------------------------------------------------------------------------------

You can improve the density-fitting accuracy by setting RI_HFX to MEDIUM/LARGE/HUGE in the CP2K input, which will increase the basis set size. In the latter case, you need to include the appropriate dimension in :code:`YOUR_SALTED_DIRECTORY/src/basis.py` and modify :code:`dfbasis` in :code:`inp.py` accordingly. To facilitate this process, go to the :code:`get_basis_info` folder and modify the CP2K input to print out the basis set information of your choice. Then, grep the RI basis and save in a file having the same name as :code:`dfbasis`, e.g., 
:code:`grep -A100 'Au  local_ri_hfx' LOCAL_BASIS_SETS > Au-DF_BASIS_NAME` 

Run the following script to generate a dictionary entry called :code:`new_basis_entry` containing the necessary information about the auxiliary basis used to generate the training data:
:code:`python get_basis_info.py`

This can be appended directly appended to the SALTED basis dictionary file:
:code:`cat new_basis_entry >> $SALTEDPATH/basis.py`

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

Calculate the spherically averaged baseline coefficients across the training set
:code:`python $SALTEDPATH/get_averages.py`

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

