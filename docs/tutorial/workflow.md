# Workflow

The general SALTED workflow is summarised below.  Detailed descriptions of each step will given in the following sections of the tutorial.

## SALTED workflow

    1. Calculate electron density and [density fitting (DF)](../Tutorial-1/README.md#density-fitting-method) coefficients.
    1. Generate (optionally sparse) [$\lambda$-SOAP descriptors](../Tutorial-1/README.md#symmetry-adapted-descriptor) by rascaline, and sparsify the atomic environments by [farthest point sampling (FPS) method](../Tutorial-1/README.md#farthest-point-sampling). This step uses the salted functions `initialize`, `sparse_selection`, and `sparse_descriptor`.
    1. Calculate [RKHS](../Tutorial-1/README.md#reproducing-kernel-hilbert-space-rkhs) related quantities, including kernel matrix $\mathbf{K}_{MM}$, associated projectors, and the feature vector $\mathbf{\Psi}_{ND}$. This step uses the salted functions `rkhs_projector` and `rkhs_vector`.
    1. Optimize [GPR weights](../Tutorial-1/README.md#gpr-optimization) by either [direct inversion](../Tutorial-1/README.md#direct-inversion) or [CG method](../Tutorial-1/README.md#conjugate-gradient-method), and save the optimized weights. This uses either the functions `hessian_matrix` and `solve_regression`, or `minimize_loss`.
    1. Validate the model using the `validation` function.
    1. Predict [density fitting coefficients](../Tutorial-1/README.md#density-fitting-method) of new structures using the [GPR weights](../Tutorial-1/README.md#gpr-optimization) obtained in the previous step, saving the predicted density coefficients. This uses the `prediction` function.
    1. Use the predicted density coefficients to calculate properties derived from the predicted electron density.

