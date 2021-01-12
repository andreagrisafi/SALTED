# system definition 
# -----------------
filename = "water_dimers_10.xyz" # XYZ file
path2ref = "../water_monomer/"
filename_ref = "water_monomers_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# QM variables 
# ------------
path2qm = "/local/big_scratch/water_dimer_asymp/density_matrices/" # path to density matrices 
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis

# RHO variables
# -------------
path2overl = "/local/big_scratch/water_dimer_asymp/overlaps/" # path to overlaps                
path2projs = "/local/big_scratch/water_dimer_asymp/projections/" # path to projections               
dfbasis = "RI-cc-pvqz" # auxiliary basis

# ML variables  
# ------------
path2soap_ref = "/local/big_scratch/water_monomer/soaps/" # path to soap features               
path2soap = "/local/big_scratch/water_dimer_asymp/soaps/" # path to soap features               
path2kern = "/local/big_scratch/water_dimer_asymp/kernels/" # path to kernels
path2preds = "/local/big_scratch/water_dimer_asymp/predictions/" # path to predictions               
z = 2.0 # kernel exponent 
Menv = 150 # number of FPS environments
