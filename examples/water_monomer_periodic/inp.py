# system definition 
# -----------------
filename = "water_monomers_1k_periodic.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# path to data
# ------------
path2data = "/scratch/grisafi/lowdin/water_monomer/" 

# RHO variables
# -------------
dfbasis = "CP2K-LRI-DZVP-MOLOPT-GTH-MEDIUM" # auxiliary basis

# ML variables  
# ------------
z = 2.0 # kernel exponent 
Menv = 50 # number of FPS environments
Ntrain = 800 # number of training structures
trainfrac = 1.0 # training set fraction
regul = 1e-10 # regularization
jitter = 1e-10 # jitter value
