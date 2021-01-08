# system definition 
# -----------------
filename = "coords_1000.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# QM variables 
# ------------
path2qm = "/local/big_scratch/rho_water_monomer/" # path to density matrices 
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis

# RHO variables
# -------------
path2overl = "./overlaps/" # path to overlaps                
path2projs = "./projections/" # path to projections               
path2preds = "./predictions/" # path to predictions               
dfbasis = "RI-cc-pvqz" # auxiliary basis

# ML variables  
# ------------
path2soap = "./soaps/" # path to soap features               
path2kern = "./kernels/" # path to kernels
z = 2.0 # kernel exponent 
Menv = 100 # number of FPS environments
Ntrain = 500 # number of training structures
trainfrac = 1.0 # training set fraction
regul = 1e-08 # regularization
jitter = 1e-10 # jitter value
