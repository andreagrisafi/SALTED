# system definition 
# -----------------
filename = "coords_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# path to data
# ------------
path2qm = "/scratch/grisafi/lowdin/liquid_water_aims/" 
path2ml = "/scratch/grisafi/lowdin/liquid_water_aims/" 
soapdir = "soap/"
kerndir = "kernels/"
preddir = "predictions/"

# RHO variables
# -------------
dfbasis = "FHI-aims-tight" # auxiliary basis
overcut = 1e-10

# ML variables  
# ------------
z = 2.0 # kernel exponent 
Menv = 2000 # number of FPS environments
Ntrain = 500 # number of training structures
trainfrac = 0.4 # training set fraction
regul = 1e-06 # regularization
eigcut = 1e-10 # eigencut
gradtol = 1e-05
restart = False
