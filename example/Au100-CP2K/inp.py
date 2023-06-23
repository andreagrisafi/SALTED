# System definition
# ---------------------------
filename = "coords.xyz" # XYZ coordinate file
species = ["Au"]             # list of species to use as centers
periodic = "2D"              # system periodicity
field = True                 # option for external field

# SALTED set up 
#--------------------------------------------------
saltedpath = "/scratchbeta/grisafia/Au-fcc100-223/"
saltedname = "rc8.0_sg0.5"         
coefdir = "coefficients-response/" # directory of density coefficients
parallel = False                   # option for MPI parallelization
average = False                    # option for spherical average baseline 

# QM variables for interface with CP2K
# ---------------------------------------------
qmcode = "cp2k"
path2qm = "/scratchbeta/grisafia/Au-fcc100-223/runs/" # path of CP2K calculations
dfbasis = "RI_AUTO_OPT-ccGRB"                         # auxiliary basis set
coeffile = "Au-RI_DENSITY_COEFFS.dat"                 # density coefficients
ovlpfile = "Au-RI_2C_INTS.fm"                         # overlap matrix
pseudocharge = 11.0                                   # pseudo nuclear charge

# Representation of 1st local environment 
# ----------------------------------------------------
rep1 = "rho" # representation kind
rcut1 = 8.0  # radial cutoff (angstrom)
sig1 = 0.5   # Gaussian width (angstrom)
nrad1 = 6    # number of radial functions 
nang1 = 6    # number of angular functions
nspe1 = 1    # number of chemical species

# Representation of 2nd local environment 
# ----------------------------------------
rep2 = "V"   # representation kind
rcut2 = 8.0  # radial cutoff (angstrom)
sig2 = 0.5   # Gaussian width (angstrom)
nrad2 = 6    # number of radial functions 
nang2 = 6    # number of angular functions
nspe2 = 1    # number of chemical species

# SALTED parameters
#----------------------------------------------------------
Menv = 50         # number of sparse atomic environments 
z = 2             # kernels exponent
Ntrain = 32       # number of training structures
trainfrac = 1.0   # training set fraction
regul = 1e-08     # regularization parameter
eigcut = 1e-10    # eigenvalues cutoff for RKHS projection
gradtol = 1e-05   # tolerance for CG minimization 
restart = False   # restart option for minimization
qscale = True     # scaling option for charge conservation
