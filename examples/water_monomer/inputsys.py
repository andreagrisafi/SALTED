# system definition in .xyz format
filename = "coords_1000.xyz"

# basis set used for the decomposition of the scalar field
basis = "RI-ccpVQZ"

# ordered list of atomic species as defined in the SOAP representations 
species = ["H","O"]

# number of atomic environments selected as sparse set
Menv = 100

# kernels non-linearity degree (1,2,...). Going beyond 2 is not recommended (default is 2)
z = 2.0
