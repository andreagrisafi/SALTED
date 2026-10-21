# Input Configuration

The file `inp.yaml` in the working directory is used to configure the input data for the model.
This file will be parsed by the `sys_utils.ParseConfig` as a dictionary.

## Input controlling file structure

This lists all possible inputs which can be listed in `inp.yaml`.
Some options may be omitted, in which case default values are chosen.
The `type` columns follow the Python typing conventions.
The `default` column lists if the variable is required or not (then it lists the default value).

For all the path-related variables, the path can be either a relative path or an absolute path.

### SALTED definition `inp.salted`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `saltedname` | `str` | **Required** | A label to identify a particular SALTED setup. |
| `saltedpath` | `str` | **Required** | Location of all files produced by SALTED. Either relative to the working directory or an absolute path. |
| `saltedtype` | `str` | `density` | Option for selecting the type of SALTED target (`density` or `density-response`) |
| `seed` | `int` | `42` | Random seed. Only implemented in parsing input; not implemented in the code. |
| `verbose` | `bool` | `False` | Output verbosity. |

### System difinition `inp.system`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `filename` | `str` | **Required** | An extended-XYZ file consisting of input structures. |
| `species` | `list[str]` | **Required** | List of element species considered in the electron density expansion. |
| `average` | `bool` | `True` | Whether we use averaged coefficients to set a baseline for the density. Normally this should be true, unless a density difference is learned. |

### Information about QM training set generation `inp.qm`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `path2qm` | `str` | **Required** | Location of the quantum-mechanical training data. |
| `qmcode` | `Literal["aims"] \| Literal["cp2k"] \| Literal["pyscf"]` | **Required** | Ab initio software used to generate training data. |
| `dfbasis` | `str` | **Required** | A label for the auxiliary basis set used to expand the density. |
| `dfbasis_file` | `str` | Optional | Path to an external basis dataset yaml file holding `dfbasis`. If unset, the `basis_data.yaml` inside to the installed salted package is used. Mainly intended for testing purposes. |
| `qmbasis` | `str` | Required if `qmcode=pyscf` | Wavefunction basis set to use when generating the training data (only for PySCF). |
| `functional` | `str` | Required if `qmcode=pyscf` | DFT functional to use when generating the training data (only for PySCF). |
| `dfmetric` | `Literal["identity"] \| Literal["coulomb"]` | Required if `qmcode=cp2k` | Metric used to perform the density fitting (only for CP2K). `coulomb` enables the calculation of the Hartree energy in `salted.cp2k.density_fitting` and `salted.validation`. |
| `periodic` | `bool` | Required if `qmcode=cp2k` | Periodic boundary conditions (only for CP2K). |

### Rascaline atomic environment parameters `inp.descriptor.rep[n]`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `type` | `Literal["rho"] \| Literal["V"]` | **Required** | Representation type, `"rho"` for atomic density and `"V"` for atomic potential. |
| `rcut` | `float` | **Required** | Radial cutoff (Angstrom) of the structural representation. |
| `nrad` | `int` | **Required** | Number of radial functions to be used for the structural representation. |
| `nang` | `int` | **Required** | Maximum angular momentum to be used for the structural representation. |
| `sig` | `float` | **Required** | Gaussian function width (Angstrom) for atomic density and/or derived atomic potential. |
| `neighspe` | `list[str]` | **Required** | List of atomic species to be used for structural representation. |

### Feature sparsification parameters `inp.descriptor.sparsify`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `nsamples` | `int` | `100` | Number of structures to use for feature sparsification. |
| `ncut` | `int` | `0` | Sets maximum number of sparse (by FPS) descriptor features to retain. `0` for no sparsification. |

### Prediction variabls `inp.predict`

Remember to set `inp.predict` if one wants to predict densities.

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `filename` | `str` | Required if predict | An extended-XYZ file consisting of structures whose densities we wish to predict. |
| `predname` | `str` | Required if predict | A label to identify a particular set of predictions. |
| `predict_data` | `str` | Required if predict and `qmcode=aims` | Path to ab initio output for prediction, relative to path2qm. |
| `alpha_only` | `bool` | `False` | Whether to limit predictions to the subset of coefficients associated with the calculation of the polarizability tensor. Only usable in combintation with `saltedtype=density-response` and `qmcode=cp2k`. |

### ML (GPR) variables `inp.gpr`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `z` | `float` | `2.0` | Kernel exponent $\zeta$. |
| `Menv` | `int` | **Required** | Number of reference environments. |
| `Ntrain` | `int` | **Required** | Number of training structures. |
| `trainfrac` | `float` | `1.0` | Training dataset fraction. Training dataset size is `Ntrain * trainfrac`. |
| `regul` | `float` | `1e-6` | Regularization parameter $\eta$. |
| `eigcut` | `float` | `1e-10` | Eigenvalues cutoff for RKHS projection. |
| `gradtol` | `float` | `1e-5` | Minimum gradient norm tolerance for CG minimization. |
| `restart` | `bool` | `False` | Whether to restart from previous minimization checkpoint. |
| `trainsel` | `Literal["sequential"] \| Literal["random"]` | `"random"` | Select the training set at random or sequentially from the entire dataset. |
| `sparse_algorithm` | `Literal["dense"] \| Literal["numba"] \| Literal["omp_sparse"]` | `"numba"` | The algorithm to compute Hessian matrices using RKHS vector sparsity. `"numba"` uses JIT-compiled kernels (default). `"omp_sparse"` uses OpenMP Fortran extensions (requires `make`; falls back to `"numba"` if unavailable). `"dense"` disables sparse operations. |
| `Mcut` | `Literal["fixed"] \| Literal["gaussian"] \| list[int]` | `"fixed"` | Per-`lam` truncation of the sparsified atomic environments used as the RKHS basis. `"fixed"` keeps all `Menv` environments at every `lam`. `"gaussian"` uses `round(Menv * exp(-0.05 * lam**2))`. `list[int]` of length `>= max(lmax)+1` gives explicit per-`lam` counts. NB: all modes share the same on-disk storage cache layout; switching modes overwrites previously generated files. |

!!! note "Tuning the Numba cache budget for Hessian computation"
    When `sparse_algorithm: numba` is used, the L3 cache budget can be controlled via the environment variable `SALTED_NUMBA_CACHE_BYTES` (default: 32 MiB). For MPI runs, set it to approximately `total_L3 / ranks_per_node / 2`:
    ```bash
    export SALTED_NUMBA_CACHE_BYTES=$(( 512*1024*1024 / 16 / 2 ))  # 512 MiB L3, 16 ranks/node
    ```

### Long-range density fitting `inp.lr` (experimental)

Settings of `salted.cp2k.LR_density_fitting` (only for CP2K), which fits the long-range part of the density, smoothed by a Gaussian of width 1 Å, on a small basis of Gaussians derived from `qm.dfbasis`. The other SALTED steps ignore this section. The outputs go to `coefficients_lr/`, `overlaps_lr/`, `density_lr/` and `metrics_lr/` in `saltedpath`.

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `recipe` | `Literal["minimal"] \| Literal["accurate"]` | `"minimal"` | LR basis recipe. `minimal`: `L <= 1` for every species, 3 widths for `l = 0` and 1 for `l = 1`. `accurate`: `L <= 2` with 3/2/1 widths, except H, which keeps `L <= 1` with 3/1. |
| `lmax` | `dict[str, int]` | Optional | Per-species override of the recipe `L_max`, e.g. `{O: 2}`. The recipe numbers of widths are truncated, or extended with their last value. |
| `nrad` | `dict[str, int \| list[int]]` | Optional | Per-species override of the recipe numbers of widths: one value for every `l`, or one per `l`, e.g. `{O: [3, 2, 1]}`. |
| `dfbasis` | `str` | Optional | Name of the LR basis, written to `saltedpath/basis/` and registered in `saltedpath/basis/lr_basis.yaml`. Default `LR-minimal` / `LR-accurate`, or a name listing the content when `lmax` or `nrad` is set (e.g. `LR-H1n31-O2n321`). Must differ from `qm.dfbasis`. |
| `from_density` | `str` | Optional | Folder of stored LR densities (`density_conf<i>.npz`, written to `density_lr/` by a previous run) to fit instead of the cube files, e.g. to compare LR bases without reading the cubes again. |
| `write_cube` | `bool` | `False` | Also write the stored LR densities as cube files. |

---

## API

For details please check the source code.

::: salted.sys_utils.ParseConfig

<br> <!-- larger space -->

---
