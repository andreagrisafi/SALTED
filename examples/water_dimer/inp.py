# system definition 
# -----------------
filename = "water_dimers_10.xyz" # XYZ file
species = ["H","O"] # ordered list of species

# reference system
# ----------------
path2ref = "../water_monomer/"
filename_ref = "water_monomers_1k.xyz" # XYZ file

# paths to data
# -------------
path2data = "/local/big_scratch/water_dimer/"
path2data_ref = "/local/big_scratch/water_monomer/"

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
