python3 -m salted.init_features
python3 -m salted.sparse_selection
mpirun -n 4 python3 -m salted.sparse_descriptors
python3 -m salted.rkhs_projector
mpirun -n 4 python3 -m salted.rkhs_vector
mpirun -n 4 python3 -m salted.matrices
python3 -m salted.regression
python3 -m salted.validation
