# Learn the Density

## Overview before starting

!!! abstract "Related starting files"
    | file or dir name | description |
    | -: | :- |
    | `README.rst` | README file for your reference |
    | `inp.yaml` | SALTED input file, consists of file paths and hyperparameters |
    | `coefficients/*` | Density fitting coefficients |
    | `projections/*` | Density projection (the $\mathbf{S}_{NN} \mathbf{c}_{N}^{DF}$) |
    | `overlaps/*` | Overlap matrix |

!!! abstract "We are going to"
    1. Sparsify the symmetry-adapted descriptors using a subset of training set (optional but often necessary for most applications).
    1. Sparsify the atomic environments to recast the learning problem in a low dimensional space. 
    1. Calculate the sparse descriptors for each reference atomic environment.
    1. Optimize GPR weights by direct inversion or conjugate gradient method.
    1. Validate the SALTED model.

These steps should be the same regardless of which electronic structure code you used to calculate the overlaps, coefficients and projections.

## Generate descriptors (with optional feature sparsification)

Currently, we have obtained the DF coefficients in dir `coefficients/` and the DF projection vectors in dir `projections/`.
To conduct the GPR, we need to generate $\lambda$-SOAP power spectra (which can be used to construct kernels) for the training dataset.
This is achieved in three steps. We first run

```bash
python -m salted.initialize
```
The following files are generated:

- `wigners/wigner_lam-[lam]_lmax1-[nang1]_lmax2-[nang2].dat`
    - The wigner $3j$ symbols.
    - In our example, the angular momentum $\lambda$ ranges from $0$ to $5$, i.e. `lam=[0..5]`. `inp.descriptor.rep1.nang` and `inp.descriptor.rep2.nang` are from `inp.yaml`.
- `equirepr_[inp.saltedname]/FEAT-0.h5`
    - The power spectrum for $\lambda=0$ is stored in an HDF5-formatted files that allow parallel read/write.

Optionally, if a `sparsify` subsection is added to the `inp.descriptor` section in `inp.yaml`, then the power spectrum for every value of $\lambda$ is FPS-sparsified along the feature axis (keeping all atoms, but selecting features). The number of retained features is specified by `inp.descriptor.sparsify.ncut`, and the number of structures used to sparsify the feature space is given by `inp.sparsify.nsamples`. It is recommended to use a fairly small fraction of your total dataset to carry out this sparsification. In this case, the files `equirepr_[inp.saltedname]/fps[inp.descriptor.sparsify.ncut]-[lam].npy` are generated which store this sparsification information.

In this example, we will not sparsify the features.

## Sparsify atomic environments

The atomic environments need to be sparsified to keep the dimensionality of the problem fixed. In this case we select `[inp.gpr.Menv]` atomic environments by running

```bash
python -m salted.sparse_selection
```

This sparsification is based on the (dense) features of angular momentum $\lambda = 0$.

The new file created is:

- `equirepr_[inp.salted.saltedname]/sparse_set_[inp.gpr.Menv].txt`
    - This file has two columns, `(feature_index, atomic_species)`.
    
## Calculate sparse descriptors

Having sparsified the atomic environments, and optionally the feature space, we then calculate the descriptors for each of the sparse environemnts, at each required value of $\lambda$, by calling

```bash
mpirun -np $ntasks python -m salted.sparse_descriptor
```

This produces the following files:

- `equirepr_[inp.salted.saltedname]/FEAT-[lam]_M-[inp.gpr.Menv].h5`
    - Features of each $\lambda$ are sliced by sparsification results above.

## Reproducing Kernel Hilbert Space (RKHS) Vectors

Having calculated the descriptors for the sparse environments, we can now calculate the kernels between each of these environments for each $\lambda$, $k_{MM}^{\lambda}$, and from those kernels find the projector matrices over the RKHS:

```bash
python -m salted.rkhs_projector
```

This produces the file `equirepr_[inp.salted.saltedname]/projector_M[inp.gpr.Menv]_zeta[inp.gpr.z].h5` which contains each of these projectors.

We then calculate the kernels across the full dataset, $k_{NM}^{\lambda}$, and project them onto the RKHS to determine the input RKHS vectors:

```bash
mpirun -np $ntasks python -m salted.rkhs_vector
```

These input vectors are stored in the files: `rkhs-vectors_[inp.salted.saltedname]/M[inp.gpr.Menv]_zeta[inp.gpr.z]/psi-nm_conf[n].npz`, where `i` runs over each structure in the training dataset.

## Regression

There are two options for us to derive the weights in GPR model:
direct inversion and loss function minimization.

### Direct inversion

!!! warning "Direct inversion is not recommended for large systems and datasets, where the dimension is greater than $10^5$! ❌"

In order to perform the solution of the problem by inversion, the following commands need to be run:

```bash
mpirun -np $ntasks python -m salted.hessian_matrix  # builds the matrices to be inverted
python -m salted.solve_regression                   # carry out matrix inversion
```

The GPR weights are stored at `regrdir_[inp.salted.saltedname]/M[inp.gpr.Menv]_zeta[inp.gpr.z]/weights_N[ntrain]_reg[log10(inp.gpr.regul)].npy`. Here `[ntrain]` is determined by `inp.gpr.Ntrain * inp.gpr.trainfrac`, and `inp.gpr.regul` is the regularization parameter.

Note that in order to parallelise the matrix inversion, we split matrices into mini-batches of size `inp.gpr.blocksize`, which must be an exact divisor of the training set size `ntrain`. `inp.gpr.blocksize` should not be included if running serially.

### Conjugate gradient method

In order to solve the problem by CG minimization, run

```bash
mpirun -np $ntasks python -m salted.minimize_loss
```

The minimization might take relatively long, and checkpoint files are stored at `regrdir_[inp.salted.saltedname]/M[inp.gpr.Menv]_zeta[inp.gpr.z]`.
The final GPR weights are stored in `regrdir_[inp.salted.saltedname]/M[inp.gpr.Menv]_zeta[inp.gpr.z]/weights_N[ntrain]_reg[log10(inp.gpr.regul)].npy`.
Part of the training dataset is sampled for training, and their indexes are written to file `regrdir_[inp.salted.saltedname]/training_set_N[inp.gpr.Ntrain].txt`.

## Validation

After obtaining the regression weights, we should validate the model by a fixed fraction of the dataset, which is not included the training structures.
This is achieved by running

```bash
mpirun -np $ntasks python -m salted.validation
```

and results are written to `validations_[inp.salted.saltedname]/M[inp.gpr.Menv]_zeta[inp.gpr.z]/N[ntrain]_ref[log10(inp.gpr.regul)]/errors.dat`,
in which the density mean absolute error is in percentage, and the average error for our example should be around 0.5%.

We can check the model performance when we vary the number of training samples (learning curves) by training several models with different values of the hyperparameter `inp.gpr.trainfrac` $\in [0.0, 1.0]$. Note that the validation set should be kept unchanged, i.e. do not change the hyperparameter `inp.gpr.Ntrain`.
