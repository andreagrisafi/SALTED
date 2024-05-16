# Workflow

The general SALTED workflow is summarised below.  Detailed descriptions of each step will given in the following sections of the tutorial.

## SALTED workflow

1. Calculate electron density and [density fitting (DF)](methods.md#density-fitting-method) coefficients.
1. Generate (optionally sparse) [$\lambda$-SOAP descriptors](methods.md#symmetry-adapted-descriptor) by rascaline, and sparsify the atomic environments by [farthest point sampling (FPS) method](methods.md#farthest-point-sampling). This step uses the salted functions `initialize`, `sparse_selection`, and `sparse_descriptor`.
1. Calculate [RKHS](methods.md#reproducing-kernel-hilbert-space-rkhs) related quantities, including kernel matrix $\mathbf{K}_{MM}$, associated projectors, and the feature vector $\mathbf{\Psi}_{ND}$. This step uses the salted functions `rkhs_projector` and `rkhs_vector`.
1. Optimize [GPR weights](methods.md#gpr-optimization) by either [direct inversion](methods.md#direct-inversion) or [CG method](methods.md#conjugate-gradient-method), and save the optimized weights. This uses either the functions `hessian_matrix` and `solve_regression`, or `minimize_loss`.
1. Validate the model using the `validation` function.
1. Predict [density fitting coefficients](methods.md#density-fitting-method) of new structures using the [GPR weights](methods.md#gpr-optimization) obtained in the previous step, saving the predicted density coefficients. This uses the `prediction` function.
1. Use the predicted density coefficients to calculate properties derived from the predicted electron density.

