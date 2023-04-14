# system definition 
# -----------------
filename = "water_monomers_1k_periodic.xyz" # XYZ file
species = ["H","O"] # ordered list of species
propname = "electro" 
parallel = False

# paths to data
# ------------
path2ml = "mldata/" 
path2qm = "qmdata/"

featdir = "feat_vecs/"
soapdir = "soaps/"
regrdir = "regr/"
kerndir = "kernels/"

coefdir = "coefficients/"
projdir = "projections/"
ovlpdir = "overlaps/"
valcdir = "validation_coeffs/"

# AIMS variables 
# ------------
dfbasis = "FHI-aims-clusters" # auxiliary basis

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 100        # number of FPS environments
Ntrain = 100      # number of training structures

trainfrac = 1.0   # training set fraction

regul = 1e-6
eigcut = 1e-10    # eigenvalues cutoff
gradtol = 1e-5    # convergence parameter
restart = False   # restart minimization

# Prediction Paths
# ------------
predict_filename = "water_dimers_10.xyz"
predict_soapdir = "predict_soaps/"
predict_kerndir = "predict_kernels/"
predict_coefdir = "predicted_coeffs/"
predict_data = "predicted_data/"
