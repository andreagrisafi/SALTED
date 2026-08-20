# Setup SALTED calculation using AIMS data
python3 -m salted.get_basis_info
python3 -m salted.aims.move_data

# Run full SALTED workflow
python3 -m salted.initialize
python3 -m salted.sparse_selection
mpirun -np 4 python3 -m salted.sparse_descriptor
python3 -m salted.rkhs_projector
mpirun -np 4 python3 -m salted.rkhs_vector

# For small dimensionality
mpirun -np 4 python3 -m salted.hessian_matrix
python3 -m salted.solve_regression

# For high dimensionality
#mpirun -np 4 python3 -m salted.minimize_loss

mpirun -np 4 python3 -m salted.validation

mpirun -np 4 python3 -m salted.prediction
