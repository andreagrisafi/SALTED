# system definition 
# -----------------
#filename = "water_monomers_1k_periodic.xyz" # XYZ file
filename = "water_dimers_10.xyz" # XYZ file
species = ["H","O"] # ordered list of species
qmcode = 'AIMS'
average = True
parallel = True
field = False
combo = False

# Rascaline atomic environment parameters
# ---------------------------------------
rep1 = 'rho'
rcut1 = 4.0
nrad1 = 8
nang1 = 6
sig1 = 0.3
rep2 = 'rho'
rcut2 = 4.0
nrad2 = 8
nang2 = 6
sig2 = 0.3
neighspe1 = ["H","O"] # ordered list of species
neighspe2 = ["H","O"] # ordered list of species

# Feature sparsification parameters
# ---------------------------------
nsamples = 150 # Number of structures to use for feature sparsification
ncut = -1 # Set ncut = -1 to skip feature sparisification

# paths to data
# -------------
saltedpath = './'
saltedname = 'serial_field_sparse_merge'

# AIMS variables 
# --------------
dfbasis = "FHI-aims-clusters" # auxiliary basis
path2qm = "qmdata/" # path to the raw AIMS output
predict_data = 'predicted_data/' # path with path2qm where derived propoerties from predicted densities will be stored

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 100        # number of FPS environments
Ntrain = 40       # number of training structures
trainfrac = 1.0   # training set fraction
regul = 1e-6      # regularisation parameter
eigcut = 1e-10    # eigenvalues cutoff

# Parameters for direct minimization
#-----------------------------------
gradtol = 1e-5    # convergence parameter
restart = False   # restart minimization

# Parameters if performing matrix inversion
#------------------------------------------
blocksize = 10

# Prediction Paths
# ------------
predname = 'prediction'
