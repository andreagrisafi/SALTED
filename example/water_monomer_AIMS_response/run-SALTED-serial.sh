# Setup SALTED calculation using AIMS data
python3 -m salted.get_basis_info

# Run full SALTED workflow
python3 -m salted.initialize
python3 -m salted.sparse_selection
python3 -m salted.sparse_descriptor
python3 -m salted.rkhs_projector
python3 -m salted.rkhs_vector

# For small dimensionality
python3 -m salted.hessian_matrix
python3 -m salted.solve_regression

# For high dimensionality
#python3 -m salted.minimize_loss

python3 -m salted.validation

python3 -m salted.prediction
