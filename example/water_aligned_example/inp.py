# system definition 
# -----------------
filename = "aligned.xyz" # XYZ file
species = ["H","O"] # ordered list of species
propname = "electro" 
parallel = True

# path to data
# ------------
path2ml = "/ada/ptmp/mpsd/alewis/water_aligned/qmdata/" 
path2qm = "/ada/ptmp/mpsd/alewis/water_aligned/qmdata" 

soapdir = "soaps_x/"
featdir = "feat_vecs_x/"
regrdir = "regr_x/"
kerndir = "kernels_x/"

ovlpdir = "overlaps/"
coefdir = "coefficients_rho1/x/"
projdir = "projections_rho1/x/"
valcdir = "validation_predictions_rho1/x/"

# QM variables 
# ------------
functional = "b3lyp" # DFT functional
qmbasis = "" # atomic basis

# RHO variables
# -------------
dfbasis = "FHI-aims-light" # auxiliary basis
overcut = 1e-08

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 3000        # number of FPS environments
Ntrain = 400      # number of training structures
trainfrac = 1.0

regul = 1e-8
eigcut = 1e-10    # eigenvalues cutoff

gradtol = 1e-4
restart = False
