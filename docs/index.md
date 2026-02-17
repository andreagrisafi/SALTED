# Welcome to SALTED documentation!

<p align="center">
  <img src="images/salted-logo.jpg" alt="Logo" style="max-width: 80%; height: auto;">
</p>


Ab initio electronic-structure methods are computationally expensive, making the calculation of electronic properties impractical for large systems and/or long simulation trajectories. In this context, the **S**ymmetry-**A**dapted **L**earning of **T**hree-dimensional **E**lectron **D**ensities (SALTED) program represents a highly transferable machine-learning method that can be used to perform inexpensive, yet accurate, predictions of the electronic charge density of a system. The transferability of the model is derived from a suitable decomposition of the electron density, which follows density-fitting, a.k.a. resolution of the identity (RI), approximations, commonly used in electronic-structure codes. In particular, we rely on a linear expansion of the electron density over a basis made of atom-centered radial functions and spherical harmonics, which can be used to represent the three-dimensional scalar field via a set of local atom-centered coefficients. From this representation of the electron density, a symmetry-adapted extension of Gaussian Process Regression is then used to perform equivariant predictions of the expansion coefficients, thus bypassing the need to learn the rotational symmetry of spherical harmonics from data.

The core ideas of the method have been first introduced in [10.1021/acscentsci.8b00551](https://pubs.acs.org/doi/10.1021/acscentsci.8b00551) and [10.1039/C9SC02696G](https://pubs.rsc.org/en/content/articlelanding/2019/sc/c9sc02696g) with applications to isolated molecules. SALTED has then been formally presented in [10.1021/acs.jctc.1c00576  ](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00576) in the context of extending the method to periodic condensed-phase systems. The current implementation of SALTED, as well as the present documentation, follows the advancements reported in [10.1021/acs.jctc.2c00850  ](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00850), where an optimization of the method is carried out by recasting the learning problem into the Reproducing Kernel Hilbert Space (RKHS).


Check out [installation](installation) to install the project. You can find out more in the [theory](theory) and [tutorial](tutorial) sections.

!!! note

    This project is under active development.

