python3 -m salted.initialize
python3 -m salted.sparse_selection
mpirun -n 4 python3 -m salted.sparse_descriptor
python3 -m salted.rkhs_projector
mpirun -n 4 python3 -m salted.rkhs_vector
mpirun -n 4 python3 -m salted.hessian_matrix
python3 -m salted.solve_regression
python3 -m salted.validation
