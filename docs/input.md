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
| `saltedtype` | `str` | `density` | Option for selecting the type of SALTED target (`density` or `density-response`)

### System difinition `inp.system`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `filename` | `str` | **Required** | An extended-XYZ file consisting of input structures. |
| `species` | `List[str]` | **Required** | List of element species considered in the electron density expansion. |
| `average` | `bool` | `True` | Whether we use averaged coefficients to set a baseline for the density. Normally this should be true, unless a density difference is learned. |
| `parallel` | `bool` | `False` | Whether to use MPI parallelization. |

### Information about QM training set generation `inp.qm`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `path2qm` | `str` | **Required** | Location of the quantum-mechanical training data. |
| `qmcode` | `Union[Literal["aims"], Literal["cp2k"], Literal["pyscf"]]` | **Required** | Which ab initio software was used to generate training data. |
| `dfbasis` | `str` | **Required** | A label for the auxiliary basis set used to expand the density. |
| `qmbasis` | `str` | Required if `qmcode=pyscf` | Wavefunction basis set to use when generating the training data (only for PySCF). |
| `functional` | `str` | Required if `qmcode=pyscf` | DFT functional to use when generating the training data (only for PySCF). |
| `pseudocharge` | `float` | Required if `qmcode=cp2k` | Pseudo nuclear charge (only for CP2K). |
| `coeffile` | `str` | Required if `qmcode=cp2k` | Density coefficients file name as printed by CP2K. |
| `ovlpfile` | `str` | Required if `qmcode=cp2k` | Overlap matrix file name as printed by CP2K. |
| `periodic` | `bool` | Required if `qmcode=cp2k` | The periodic boundary conditions (only for CP2K). |

### Rascaline atomic environment parameters `inp.descriptor.rep[n]`

| var name | type | default | usage |
| -:| :-: | :-: | :- |
| `type` | `Union[Literal["rho"], Literal["V"]]` | **Required** | Representation type, `"rho"` for atomic density and `"V"` for atomic potential. |
| `rcut` | `float` | **Required** | Radial cutoff (Angstrom) of the structural representation. |
| `nrad` | `int` | **Required** | Number of radial functions to be used for the structural representation. |
| `nang` | `int` | **Required** | Maximum angular momentum to be used for the structural representation. |
| `sig` | `float` | **Required** | Gaussian function width (Angstrom) for atomic density and/or derived atomic potential. |
| `neighspe` | `List[str]` | **Required** | List of atomic species to be used for structural representation. |

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
| `blocksize` | `int` | `0` | Divide dataset into blocks with blocksize for MPI matrix inversion. |
| `trainsel` | `Union[Literal["sequential"], Literal["random"]]` | `"random"` | Select the training set at random or sequentially from the entire dataset. |

---

## API

For details please check the source code.

::: salted.sys_utils.ParseConfig

<br> <!-- larger space -->

---
