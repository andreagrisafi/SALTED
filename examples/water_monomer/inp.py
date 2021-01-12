# system definition 
# -----------------
filename = "coords_1000.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# QM variables 
# ------------
path2qm = "/local/big_scratch/water_monomer/density_matrices/" # path to density matrices 
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis

# RHO variables
# -------------
path2overl = "/local/big_scratch/water_monomer/overlaps/" # path to overlaps                
path2projs = "/local/big_scratch/water_monomer/projections/" # path to projections               
dfbasis = "RI-cc-pvqz" # auxiliary basis

# ML variables  
# ------------
path2soap = "/local/big_scratch/water_monomer/soaps/" # path to soap features               
path2kern = "/local/big_scratch/water_monomer/kernels/" # path to kernels
path2preds = "/local/big_scratch/water_monomer/predictions/" # path to predictions               
z = 2.0 # kernel exponent 
Menv = 100 # number of FPS environments
Ntrain = 500 # number of training structures
trainfrac = 1.0 # training set fraction
regul = 1e-08 # regularization
jitter = 1e-10 # jitter value
