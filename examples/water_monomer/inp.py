# system definition in .xyz format
filename = "coords_1000.xyz"

# basis set for the scalar-field decomposition 
basis = "RI-ccpVQZ"

# valence-ordered list of atomic species 
species = ["H","O"]

# number of atomic sparse environments 
Menv = 100

# number of training configurations 
Ntrain = 200

# training set fraction
trainfrac = 1.0

# regularization
regul = 1e-08

# jitter value
jitter = 1e-10

# kernel non-linearity degree (1,2,...)
z = 2.0
