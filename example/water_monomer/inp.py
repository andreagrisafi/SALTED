# system definition 
# -----------------
filename = "water_monomers_1k.xyz" # XYZ file
species = ["H","O"] # ordered list of species
propname = "electro" 

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
preddir = "predictions/"

# PySCF variables 
# ------------
functional = "b3lyp" # DFT functional
qmbasis = "cc-pvqz" # atomic basis
dfbasis = "RI-cc-pvqz" # auxiliary basis
overcut = 1e-08

# ML variables  
# ------------
z = 2.0           # kernel exponent 
Menv = 100        # number of FPS environments
Ntrain = 100      # number of training structures

trainfrac = 1.0   # training set fraction

regul = 1e-9      # regularization
eigcut = 1e-10    # eigenvalues cutoff
gradtol = 1e-5    # convergence parameter
restart = False   # restart minimization

# Prediction Paths
# ------------
predict_filename = "water_dimers_10.xyz"
predict_soapdir = "predict_soaps1/"
predict_kerndir = "predict_kernels1/"
