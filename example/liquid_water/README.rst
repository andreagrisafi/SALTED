Compute descriptors per basis function type for a given training set
:code:`python ../../src/rkhs.py`

Compute global feature vector and save as sparse object 
:code:`python ../../src/feature_vector.py`

Minimize loss function and print out regression weights
:code:`srun -n $ntasks python ../../minimize_loss-parallel.py` 

Validate model predicting on the remaining structures
:code:`python ../../validation.py` 
