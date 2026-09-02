---
title: 'SALTED: A symmetry-adapted machine-learning program for predicting electron-densities in molecules and materials'
tags:
  - Python
  - machine learning
  - electron density
  - Gaussian process regression
  - electronic structure
  - density-functional theory
authors:
  - name: Zekun Lou
    orcid: 0009-0009-7792-3202
    affiliation: 1
  - name: Alan M. Lewis
    orcid: 0000-0002-3296-7203
    affiliation: 2
  - name: Théophane Bernhard
    orcid: 0009-0001-0130-5012
    affiliation: 5
  - name: Lukas Seifert
    orcid: 0009-0009-4032-058X
    affiliation: 3
  - name: Agustin Salcedo
    orcid: 0000-0001-5525-8605
    affiliation: 6
  - name: Florian Kleemiss
    orcid: 0000-0002-3631-1535
    affiliation: 3
  - name: Mariana Rossi
    orcid: 0000-0002-3552-0677
    affiliation: "1, 4"
  - name: Andrea Grisafi
    corresponding: true
    orcid: 0000-0003-1433-125X
    affiliation: 5
affiliations:
  - name: MPI for the Structure and Dynamics of Matter, Hamburg, Germany
    index: 1
  - name: Department of Chemistry, University of York, York, UK
    index: 2
  - name: Institute of Inorganic Chemistry, RWTH Aachen University, Landoltweg 1a, 52074 Aachen, Germany
    index: 3
  - name: Yusuf Hamied Department of Chemistry, Cambridge University, Cambridge, UK
    index: 4
  - name: Physicochimie des Électrolytes et Nanosystèmes Interfaciaux, Sorbonne Université, CNRS, F-75005 Paris, France
    index: 5
  - name: Laboratoire de physique de L’École normale supérieure de Paris, CNRS, ENS & Université PSL, Sorbonne Université, Université de Paris, F-75005 Paris, France
    index: 6
date: 10 September 2026
bibliography: paper.bib
---

# Summary

`SALTED` provides an open-source Python package for machine learning the quantum-mechanical electron density, $n(\mathbf{r})$, in molecular and condensed-phase systems based on input atomic coordinates and species [@gris+19acscs; @lewis+21jctc; @Grisafi2023]. The program adopts a linear atom-centered decomposition of the electron density, which makes it highly transferable across diverse atomistic configurations sharing similar chemical environments. Because of this representation choice, `SALTED` is naturally interfaced with state-of-the-art electronic-structure programs based on atomic orbitals, namely CP2K [@cp2k-made-easy-2026], FHI-aims [@roadmap-aims-2026], and PySCF [@PySCF2020], from which reference electron-density data can be generated and used to train a model. The learning algorithm is based on a symmetry-adapted extension of Gaussian process regression [@gris+18prl], making `SALTED` especially efficient in small-data regimes. Thanks to the implementation of vector-field kernel functions [@Rossi2025], `SALTED` can also learn the first-order response of the electron density to applied electric fields, $\partial n(\mathbf{r})/\partial \mathbf{E}$. The application of `SALTED` within computational workflows has already shown its utility in a wide variety of contexts, including the calculation of polarization vectors [@grisafi2023prm] and polarizability tensors [@Rossi2025], the accurate evaluation of Coulomb forces in QM/MM molecular-dynamics simulations [@Grisafi2024], and electronic-structure studies of large-scale 2D materials [@lou2026prx].

# Statement of need

Electronic structure methods can be predictive for the simulation of the properties of a wide variety of materials. Among these methods, density-functional theory (DFT) has, perhaps, become the most successful and popular, having solved outstanding problems in areas as diverse as biochemistry, catalysis, nanotechnology and quantum materials [@Burke2012persp; @Jones2015rmp]. The popularity of DFT is based on its ability to provide sufficient accuracy for many practical applications at a modest cost. However, the cost of these calculations can only be considered modest in comparison to other, more advanced, electronic structure methods. For the current landscape of data-driven material science and molecular discovery, allied to the quest of achieving first-principles accuracy for larger systems and longer time scales, even the cost of DFT calculations becomes prohibitively large.

A machine-learning method that can predict the most fundamental quantity of DFT, namely the real-space electronic density, therefore holds immense potential to deliver a single model that can be used to calculate a multitude of downstream material properties at a small fraction of the cost of DFT, while maintaining very similar accuracy. In this paper, we describe the `SALTED` software package, which allows electronic density and electronic-density response predictions over an atomic basis consistent with the underlying electronic-structure architectures.

The availability of this open-source software package provides the atomistic-simulation community with a practical and extensible framework for constructing machine-learning models of the electron density and its linear response from first-principles reference data. `SALTED` is intended for researchers in computational chemistry, condensed-matter physics, materials science, and molecular simulation who wish to accelerate electronic-structure calculations while retaining direct access to physically meaningful electronic observables. By interfacing with widely used electronic-structure packages, `SALTED` can be embedded into established computational workflows. The package is designed both for users seeking efficient predictions of electron densities and derived properties, and for method developers interested in extending the methodology or in interfacing density predictions with downstream electronic-structure and multiscale simulation tools. `SALTED` facilitates reproducible research and supports the development of machine-learning models for electronic structure by providing an open, documented, and reusable implementation of electron-density prediction.

# State of the field

Two main directions have been independently followed to construct ML models of the electron density that either directly sample $n(\mathbf{r})$ on a real-space 3D grid, or expand $n(\mathbf{r})$ over a linear atomic basis. Examples of grid-based ML models with published open-source packages are: `DeepDFT`, based on equivariant graph neural networks [@Jorgensen2022]; a linear-regression framework based on Jacobi–Legendre many-body descriptors [@Focassio2023]; `Charge3net`, based on a high-order equivariant neural network [@Koker2024]; and an extension of `FIREANN`, where an efficient sampling strategy is devised to drastically reduce the number of grid-point evaluations and to further treat the response to applied fields [@Feng2025]. Albeit not targeting directly the electronic density, the Materials Learning Algorithm (`MALA`) package learns the local density of states on a grid, and reconstructs $n(\mathbf{r})$ from this quantity [@Cangi_MALA_2025].

When compared with grid-based approaches, ML models that represent the density on an atomic basis carry the advantage of greatly reducing the amount of data at the price of an acceptably small error due to the finite expressiveness of the basis functions. The `SALTED` package is the result of one of the earliest ML approaches developed to learn the electron density via a set of atomic coefficients [@gris+19acscs]. Specifically, $n(\mathbf{r})$ is expanded over a linear basis made of radial functions $R^{\lambda}_{n}$ and spherical harmonics $Y_{\mu}^{\lambda}$ centered around each atom of the system:

$$n(\mathbf{r}) = \sum_{in\lambda\mu} c_{in\lambda\mu} \sum_{\mathbf{u}} R^\lambda_{n}\left(\left|\mathbf{r}-\mathbf{r_{i}} -\mathbf{u}\right|\right)\, Y_{\mu}^{\lambda}\left(\widehat{\mathbf{r}-\mathbf{r_{i}}-\mathbf{u}}\right)$$

where $i$ are the atomic indexes, $\mathbf{u}$ are the cell translation vector when considering periodic systems [@lewis+21jctc], and $c_{in\lambda\mu}$ are the density expansion coefficients. The reference data for $c_{in\lambda\mu}$ can directly be obtained from resolution of the identity (RI) methods implemented in electronic-structure codes that are based on atomic orbitals, making `SALTED` easy to interface with codes using such basis sets and methods.

As a key difference from the models previously mentioned, `SALTED` is based on a symmetry-adapted extension of Gaussian process regression (GPR). Spherical-tensor kernels [@gris+18prl], $\mathbf{K}^\lambda$, rather than equivariant neural-networks, are computed to satisfy exact $\mathcal{O}(3)$ symmetries [@gris+19acscs], starting from atom-density features obtained using the `featomic` library [@bigi_metatensor_2026]. While the adoption of a GPR method limits the scalability of `SALTED` over strongly heterogeneous datasets [@Grisafi2023], the constrained function space defined by the kernels, together with the physically interpretable nature of the underlying atomistic features, helps avoid overfitting—thereby facilitating transferable extrapolations of the electron density across different system sizes. Moreover, when compared with data-hungry neural-network architectures, the data-efficiency of GPR-based approaches becomes a great advantage when training the model on high-level electronic-structure methods, for which only a small number of calculations ($\sim 10^{2-3}$) can be performed.

In addition, `SALTED` presents a couple of distinctive features. First, a suitable linear combination of the kernels $\mathbf{K}^\lambda$ allows `SALTED` to learn the first-order response of the density to uniform applied electric fields, $\partial n(\mathbf{r})/\partial E_k$, thereby ensuring the correct transformations of a 3D vector field [@Rossi2025].
Second, the use of the `featomic` library enables the inclusion of long-distance equivariants (LODE) features [@grisafi2019jcp], which are essential for learning the nonlocal redistribution of $n(\mathbf{r})$ in highly-polarizable systems such as metallic frameworks [@grisafi2023prm; @Rossi2025].

We note that there are other ML packages that share the same philosophy as `SALTED` in representing the electron density similarly to the equation above, albeit with different features and capabilities: equivariant neural-network architectures based on the `E3NN` model [@Rackers2023]; `scdp`, which augments the atom-centered basis with additional "virtual" orbitals placed at non-atomic sites (e.g., bond midpoints) to improve expressivity, and regresses the expansion coefficients with a high-capacity equivariant neural network [@fu2024recipe]; and lastly, an equivariant neural-network architecture that predicts atom-centered density coefficients as an intermediate representation for learning energies and forces [@bogojeski2026].

# Software design

`SALTED` is organized as a workflow in three stages: dataset preparation, model training, and prediction (\autoref{fig:workflow}).
Each stage is configured through a dedicated section of a single `inp.yaml` file, and each step within a stage is exposed as an independently invocable command, `python -m salted.<step>`.
GitHub Continuous Integration (CI) is enabled for testing the package: unit tests cover the utility modules, and pipeline tests run the full workflow through every electronic-structure interface.

![**SALTED workflow**. Reference density or density-response RI coefficients $\mathbf{c}^\mathrm{RI}$ of the training dataset are generated by CP2K, FHI-aims, or PySCF. Model preparation builds $\lambda$-descriptors and assembles kernels $\mathbf{K}^{\lambda}$ of the training structures, with optional sparsification. Model training obtains regression weights $\mathbf{w}^\lambda$ using $\mathbf{c}^\mathrm{RI}$ and $\mathbf{K}^\lambda$. The trained models can be quickly validated to determine their accuracy, before being used to evaluate unseen structures. Predicted RI coefficients $\mathbf{c}^\mathrm{ML}$ are used either directly, to evaluate electrostatic properties such as multipoles, electric fields, and polarizabilities, or as an input density from which the Kohn--Sham Hamiltonian is built and further obtain DFT properties after one diagonalization step. Shaded boxes mark stages that are parallelized by MPI. \label{fig:workflow}](figures/workflow.png)

This step-wise organization allows a run to be check-pointed, restarted, and distributed across separate HPC allocations, with the intermediate quantities of each step available on disk.
Such control is necessary in practice: training a capable model over a large chemical space or on condensed-phase systems usually requires many HPC nodes, and the most efficient parallelization strategy differs from step to step.
Users can resume an interrupted run, reuse descriptors across hyperparameter choices, and inspect intermediate quantities to improve models.
We accept a less immediate interactive experience that a single-call library API would provide in exchange for workflows that remain manageable by the user.

`SALTED` integrates with a variety of electronic-structure codes based on atomic orbitals, CP2K [@cp2k-made-easy-2026], FHI-aims [@roadmap-aims-2026], and PySCF [@PySCF2020], thus supporting a large fraction of community-developed electronic-structure software. Density data from CP2K are provided on uniform grids as `*.cube` files following pseudo-valence representations of $n(\mathbf{r})$; `SALTED` can then compute the required RI coefficients via a dedicated `density_fitting` module. 
FHI-aims is an all-electron electronic-structure program; in this case, density data for isolated and periodic systems are directly obtained as RI coefficients on the same footing. Finally, pseudo-valence PySCF density data for finite molecular systems can be obtained from `SALTED` by calling relevant PySCF functions.

`SALTED` unifies the training data obtained from these different codes within a single shared format.
This requires extra engineering in the interfaces, but it lets research groups keep the electronic-structure code they already use in production, together with its established settings, and integrate `SALTED` into an existing workflow rather than switching electronic-structure codes. CI tests cover all the interfaces. We note that both CP2K and PySCF work with Gaussian-type orbitals, enabling straightforward analytical calculations of density and density-response derived properties from the predicted `SALTED` coefficients, such as Hartree energies, electric-fields, dipoles and polarizabilities. While some of these properties are already computed and printed by `SALTED` upon density prediction, others could readily be implemented as dedicated post-processing modules.

The performance-critical steps, including equivariant descriptor contraction and the construction of Hessian matrix, are implemented as `numba`-compiled Python kernels [@numba2015].
The `numba` kernels are carefully composed and benchmarked, and their portability does not come at the cost of performance.
MPI parallelization is optional: parallelizable steps run serially by default and use `mpi4py` [@mpi4py2021] automatically when launched under `mpirun`.
OpenMP and Python thread pools would parallelize these steps within a single shared-memory node, whereas MPI covers the full range of hardware of the users, from personal computers and small workstations to multi-node HPC clusters. To improve portability, the supplied Dockerfile can be used to build a self-contained image with an MPI-enabled version of `SALTED`.

# Research impact statement

The integration of `SALTED` predictions within computational workflows has already been shown useful in real-case scientific applications. For example, `SALTED` predictions of the charge-density response in metallic electrodes could be used to achieve a $\sim10^3$ speedup in the calculation of DFT-level electrostatic forces entering the quantum-mechanics/molecular-mechanics simulation of model ionic capacitors [@Grisafi2023]. More recently, the use of the predicted density as a DFT input to perform a single Kohn-Sham Hamiltonian diagonalization has enabled the study of the electronic properties of twisted bilayer Moiré materials over extremely large supercells [@lou2026prx]. Beyond electrostatic and DFT properties, `SALTED` enables the calculation of electron-density-based features at scale. A relevant example is the application to quantum-chemistry molecular interaction analysis [@fabr+20chimia]. Additionally, the RI density coefficients are directly applicable to crystal structure refinement from X-ray and electron diffraction data [@seifert2026], making the integration of `SALTED` predictions within these algorithms highly promising.

# AI usage disclosure

We declare that no AI tool was used to make core code design decisions, nor were they used to make scientific decisions.
Generative AI tools were employed to help with code refactoring and cleaning and with text polishing.

# Acknowledgements

We acknowledge funding from the French National Research Agency under the France 2030 program (PEPR BATMAN, Grant No. ANR-22-PEBA-0002 and PEPR DIADEM, Grant No. ANR-22-PEXD-0001) and the German Research Foundation (DFG) Project-ID 555467911 - CRC 1772 / TP A06.

# References
