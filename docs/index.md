# Welcome to SALTED's documentation!

Ab initio electronic structure methods, such as density functional theory, are computationally costly, rendering them impractical for large systems.
In contrast, data-driven machine learning methods represent a good alternative for predicting the electornic structure at a low computational cost,
allowing for "ab initio accuracy" in large systems.

In this context, the **Symmetry-Adapted Learning of Three-Dimensional Electron Densities** (SALTED) method [10.1021/acs.jctc.2c00850](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00850) is designed to predict electron densities in clusters and periodic systems using a locally and symmetry-adapted representation of the density field.
The approach employs a resolution of the identity, or density fitting (DF), method to project electron densities onto a new basis sets formed by prducts of atomic orbitals (AOs).
Then the Symmetry-Adapted Gaussian Process Regression (SA-GPR) method is applied to learn density fitting coefficients from a dataset comprising smaller systems, showing good capabilities of extrapolating to larger systems.


Check out [installation](installation) to install the project, and you can find more in the [tutorial](tutorial) section.

!!! note

    This project is under active development.




## How to locally setup the MkDocs website

Please install `mkdocs` and packages in `docs/requirements.txt`:

```bash
python -m pip install mkdocs
python -m pip install -r ${PROJECT_ROOT}/docs/requirements.txt
```

Then, you can run the following command to build the website:

```bash
mkdocs serve
```

Mkdocs now builds the website and makes it available under localhost:8000 (you can just enter this in your browser). The port might be a different on your platform, but mkdocs tells you where to find the website like this:

```text
Serving on http://127.0.0.1:8000/tutorials/fhi-aims-with-salted/
```

Mkdocs now watches your changes and rebuilds whenever you change something in the files (you need to save your changes). So you can watch your changes live in the browser.


