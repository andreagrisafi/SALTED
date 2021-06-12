###################################################################
################## UNCERTAINTY ESTIMATION SCRIPTS #################
###################################################################

These scripts assume that you have already built power spectra and kernels for a full training set. In order to subsample in this set, use the command:

$ subsample.py -k K_NM.npy -f file.xyz -np NP -ns NS

Here, file.xyz is the file containing the coordinates of the training set, and K_NM.npy is the kernel between members of this set and the active set. NP is the number of points in each subsample, and NS is the number of subsamples required.

The next step is to train NS models, for which the script multi_train.sh can be used:

$ multi_train.sh RANK REG K_MM.npy

RANK is the rank of the spherical tensor you want to train (full tensors not yet possible), REG the regularization and K_MM.npy the kernel between members of the active set. This script also deletes all of the intermediate FRAMES.*.xyz and KERNEL.*.xyz files created by subsample.py, in order to save space; this can be commented out if not required.

The next step is to make predictions for members of a validation set. For this, a script like multi_predict.sh is needed (this currently has rank-2 tensors hard-coded, and should be modified as desired; please see comments in this script for further information).

Finally, we calibrate the factor used to scale the uncertainty. The script get_alpha.sh creates a file, alpha.txt, for which the final line is the factor by which the squared error should be scaled (i.e., take the square root of this number to give the factor by which the prediction should be scaled about the mean).
