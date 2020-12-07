# system definition in .xyz format
filename = "coords_training.xyz"
filename_testing = "coords_testing.xyz"

# path to soaps 
dirsoap = "./training/soaps/"                
dirsoap_testing = "./testing/soaps/"                

# path to kernels
dirkern = "./training/kernels/"                
dirkern_testing = "./testing/kernels/"                

# path to overlaps
dirover = "./training/overlaps/"                
dirover_testing = "./testing/overlaps/"                

# path to projections
dirprojs = "./training/projections/"                
dirprojs_testing = "./testing/projections/"                

# basis set for the scalar-field decomposition 
basis = "RI-ccpVQZ"

# valence-ordered list of atomic species 
species = ["H","O"]

# number of atomic sparse environments 
Menv = 100

# number of training configurations 
Ntrain = 250

# training set fraction
trainfrac = 1.0

# regularization
regul = 1e-08

# jitter value
jitter = 1e-10

# kernel non-linearity degree (1,2,...)
z = 2.0
