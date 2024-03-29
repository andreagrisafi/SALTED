python3 -m salted.init_features
python3 -m salted.sparse_selection
mpirun -n 8 python3 -m salted.sparse_vector
python3 -m salted.rkhs_projector
mpirun -n 8 python3 -m salted.rkhs_vector
python3 -m salted.matrices
python3 -m salted.regression
python3 -m salted.validation
