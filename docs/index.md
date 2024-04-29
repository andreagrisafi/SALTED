{!README.md!}

# Welcome to SALTED's documentation!

Ab initio electronic structure methods, such as density functional theory, are computationally costly, rendering them impractical for large systems.
In contrast, data-driven machine learning methods represent a good alternative for predicting the electornic structure at a low computational cost,
allowing for "ab initio accuracy" in large systems.

In this context, the **Symmetry-Adapted Learning of Three-Dimensional Electron Densities** (SALTED) method [10.1021/acs.jctc.1c00576  ](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00576) is designed to predict electron densities in periodic systems using a locally and symmetry-adapted representation of the density field.
The approach employs a resolution of the identity, or density fitting (DF), method to project electron densities onto a new basis sets formed by prducts of numeric atomic orbitals (NAO).
Then the Symmetry-Adapted Gaussian Process Regression (SAGPR) method is applied to learn density fitting coefficients from a dataset comprising smaller systems, showing good capabilities of extrapolating to larger systems.


Check out [installation](installation) to install the project, and you can find more in the [tutorial](tutorial) section.

!!! note

    This project is under active development.

