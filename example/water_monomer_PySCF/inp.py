# system definition 
# -----------------
filename = "water_monomers_1k.xyz" # XYZ file
#filename = "water_dimers_10.xyz" # XYZ file
species = ["H","O"] # ordered list of species
qmcode = 'pyscf'
average = True
parallel = False 
field = False

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
sparsify = False 
nsamples = 100 # Number of structures to use for feature sparsification
ncut = 1000 # Set ncut = 0 to skip feature sparisification

# paths to data
# -------------
saltedpath = './'
saltedname = 'test'

# AIMS variables 
# --------------
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis
dfbasis = "RI-cc-pvqz" # auxiliary basis

path2qm = "./" # path to the raw AIMS output

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 10        # number of FPS environments
Ntrain = 800       # number of training structures
trainfrac = 1.0   # training set fraction
regul = 1e-10      # regularisation parameter
eigcut = 1e-10    # eigenvalues cutoff

# Parameters for direct minimization
#-----------------------------------
gradtol = 1e-5    # convergence parameter
restart = False   # restart minimization

# Parameters if performing matrix inversion
#------------------------------------------
blocksize = 0
trainsel = 'random'

# Prediction Paths
# ------------
predname = 'dimer'
