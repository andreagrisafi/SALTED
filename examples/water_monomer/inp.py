# system definition 
# -----------------
filename = "water_monomers_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species
propname = "electro" 

# path to data
# ------------
path2ml = "/local/big_scratch/water_monomer/" 
path2qm = "/local/big_scratch/water_monomer/" 
soapdir = "soaps/"
kerndir = "kernels/"
preddir = "predictions/"

# QM variables 
# ------------
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis

# RHO variables
# -------------
dfbasis = "RI-cc-pvqz" # auxiliary basis
overcut = 1e-08

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 100        # number of FPS environments
Ntrain = 100      # number of training structures
trainfrac = 1.0   # training set fraction
regul = 1e-10     # regularization
eigcut = 1e-10    # eigenvalues cutoff

xv = False
svd = False
jitter = 1e-10    # jitter value
