# system definition 
# -----------------
filename = "water_monomers_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species
propname = "electro" 

# path to data
# ------------
path2data = "/local/big_scratch/water_monomer/" 
path2indata = "/local/big_scratch/water_monomer/" 

# QM variables 
# ------------
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis

# RHO variables
# -------------
dfbasis = "RI-cc-pvqz" # auxiliary basis

# ML variables  
# ------------
z = 2.0 # kernel exponent 
Menv = 100 # number of FPS environments
Ntrain = 200 # number of training structures
trainfrac = 1.0 # training set fraction
regul = 1e-10 # regularization
jitter = 1e-10 # jitter value
xv = False
svd = False
