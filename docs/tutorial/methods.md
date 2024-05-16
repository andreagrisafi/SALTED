# Methods

This part of the tutorial contains no hands-on instructions. Instead, it is meant as a primer of the theoretical background and workflow of SALTED, for users to get familiarized with the methods before doing the exercise.


### Symmetry-adapted descriptor

<!--Different from the general GPR formalism, the atomic environment descriptor strategy is introduced to represent the atomic environment quantitatively,
and the kernel function is defined by the outer product of two (sets of) descriptor's power spectrums.-->

SALTED uses symmetry-adapted Gaussian process regression (SAGPR) to learn the density fitting coefficients. This employs $\lambda$-SOAP descriptors, modified versions of the SOAP descriptors which builds rotational equivariance into the kernel function and the descriptor's power spectrum.
For details, please check [this paper](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002  ).


### Density fitting method

The density fitting (DF) method or resolution of the identity (RI) ansatz, are commonly used in quantum chemistry. Several metrics can be chosen to perform this fitting.
It assumes that the electron density $\rho(\mathbf{r})$ could be expanded as a linear combination of products of atomic orbitals (AO) in a given metric. Here we use the Coulomb
metric and numeric AOs, which are also used in all hybrid-functional and Hartree-Fock calculations in FHI-aims.

$$
\begin{aligned}
\rho(\mathbf{r}) \approx \tilde{\rho}(\mathbf{r})
&= \sum\limits_{i,\sigma,\mathbf{U}} c_{i,\sigma} \phi_{i,\sigma} (\mathbf{r} - \mathbf{R}_{i} + \mathbf{T}(\mathbf{U})) \\
&= \sum\limits_{i,\sigma,\mathbf{U}} c_{i,\sigma} \left\langle\mathbf{r} \middle| \phi_{i,\sigma}(\mathbf{U}) \right\rangle \\
\end{aligned}
$$

where $i$ indicates the index of the center atom of the product basis, $\sigma = n \lambda \mu$ is a composite index with $n$ corresponding to the principal quantum number and $\lambda \mu$ corresponding to the spherical harmonics $Y_{\lambda}^{\mu}$.
The density fitting coefficients are also written as $\mathbf{c}^{DF}$ for distinction.

Within the GPR formalism, the density coefficients $c_{i, n\lambda\mu}$ are expressed as a linear combination of the kernel functions

$$
c_{i, n\lambda\mu} = c_{n\lambda\mu} (A_{i}) \approx \sum\limits_{j \in M} \sum\limits_{|\mu'| \le \lambda}
b_{n\lambda\mu'} (M_{j}) k_{\mu\mu'}^{\lambda}(A_{i},M_{j}) \delta_{a_{i}, a_{j}}
$$

where $b_{n\lambda\mu'}$ are the new unknown coefficients,
$A_{i}$ is the atomic environment of atom $i$,
$M_{j}$ is the subsampled atomic environment of atom $j$ in the training dataset (described in the [next section](#farthest-point-sampling)),
$k_{\mu\mu'}^{\lambda}(A_{i},M_{j})$ is the kernel function between atom $i$ and atom $j$ and indexed by $\mu\mu'$,
and $\delta_{a_{i}, a_{j}}$ is the Kronecker delta function that makes sure the atomic species of atom $i$ and atom $j$ are the same. Note that the kernel functions are built over a sparse (and diverse) set of atomic environments, which is obtained by farthest point sampling (FPS).

<!-- So far, the training dataset consists of *ab initio* calculations, density fitting coefficients ($\mathbf{c}^{DF}$), and atomic environment kernel functions $\mathbf{K}$. -->

<!--The most important term $b_{n\lambda\mu'} (M_{j})$ is the GPR weights we wish to determine by the training dataset,
while the atomic environments $M_{j}$ in the training dataset might lead to a huge kernel matrix $\mathbf{K}_{MM}$ ($M$ samples in training dataset). Proper subsampling or sparsification is necessary to reduce the computational cost, and the farthest point sampling (FPS) method is used in SALTED. -->


### Farthest point sampling

FPS is a (simple but useful) greedy algorithm to select a diverse subset of points from a given set.
Please check, e.g., this [blog](https://minibatchai.com/2021/08/07/FPS.html) for details.

In SALTED, the FPS algorithm is used to sparsify the atomic environments in two ways: (i) It is used on the feature vector dimension ($\lambda$-SOAP) itself, to select representative entries of the feature vector, and (ii) on the atomic environments, to select representative but diverse atomic environments, based on their pre-sparsified feature vector.
The theory behind FPS is the subset of regressors approach (SoR),
i.e. approximating the full kernel matrix ($N$ samples) by a subset of kernel matrix ($M$ samples, $M < N$)

$$
K_{NN} \approx K_{NM} K_{MM}^{-1} K_{MN}
$$


It is worth noting that the FPS is performed within each $an\lambda$ channel based on the $\lambda$-SOAP descriptor formalism, where $a$ is the atomic species, $n$ is the radial basis index, and $\lambda$ is the angular momentum.


### Reproducing kernel Hilbert space (RKHS)

Even though the sparsification simplifies the kernel matrix, it still involves calculating  $\mathbf{K}_{MM}^{-1}$, which can be both expensive and ill-conditioned.

Following the [representer theorem](https://en.wikipedia.org/wiki/Representer_theorem) and assuming we have a positive-definite kernel (which is the case, also after sparsification), 
we are guaranteed to find a Hilbert space with elements $\mathbf{\Psi}$ whose dot product reproduces the kernel. This is called the reproducing kernel Hilbert space (RKHS).  

We use the RKHS here to avoid the numerical hurdles which could be introduced by calculating  $\mathbf{K}_{MM}^{-1}$. First, we perform an SVD decomposition of $\mathbf{K}_{MM}^{an\lambda}$ (in channel $an\lambda$). We then keep only the non-negligible
eigenvalues $\lambda$ (larger than a cutoff threshold $\epsilon$) and their respective eigenvectors  $\mathbf{v}$. We then approximate $\mathbf{K}_{NN}^{an\lambda}$ by


$$
\mathbf{K}_{NN}^{an\lambda} \approx \mathbf{K}_{NM}^{an\lambda} \sum_d^{D^{an\lambda}} \frac{\mathbf{v}_d^M (\mathbf{v}_d^M)^T}{\lambda_d}  \mathbf{K}_{MN}^{an\lambda}
$$

$$
\mathbf{K}_{MM}^{an\lambda}
\approx \sum\limits_{d}^{D^{an\lambda}} \mathbf{v}_{M}^{d} \lambda_{d}^{-1} (\mathbf{v}_{M}^{d})^{\top}
= \mathbf{V}_{MD}^{an\lambda} (\mathbf{\Lambda}_{DD}^{an\lambda})^{-1} (\mathbf{V}_{MD}^{an\lambda})^{\top}
$$

where $D^{an\lambda}$ is the final truncated dimension, $\mathbf{\Lambda}_{DD}^{an\lambda}$ is the diagonal matrix of selected eigenvalues, and the $\mathbf{V}_{MD}^{an\lambda}$ is the batched selected eigenvectors.
With this expression, we can define the RKHS of  $\mathbf{K}_{NN}^{an\lambda}$ spanned by the feature vectors $\mathbf{\Psi}_{ND}^{an\lambda}$

$$
\begin{aligned}
\mathbf{\Psi}_{ND}^{an\lambda} \equiv & \mathbf{K}_{NM}^{an\lambda} \mathbf{V}_{MD}^{an\lambda} (\mathbf{\Lambda}_{DD}^{an\lambda})^{-1/2} \\
\end{aligned}
$$

which yields (as expected)

$$
\mathbf{K}_{NN}^{an\lambda} \approx \mathbf{\Psi}_{ND}^{an\lambda} (\mathbf{\Psi}_{ND}^{an\lambda})^{\top}
$$

With the RKHS reformulation above, we can reformulate the density-learning problem as a linear regression task parametrized according to the feature vectors $\mathbf{\Psi}_{ND}^{an\lambda}$.
Predicting density coefficients $c_{an\lambda\mu} (A_i)$ in the [previous section](#density-fitting-method) is rewritten as

$$
\begin{aligned}
c_{an\lambda\mu} (A_i)
& \approx \sum\limits_{j \in N} \sum\limits_{|\mu'| \le \lambda}
b_{n\lambda\mu'} (M_{j}) k_{\mu\mu'}^{\lambda}(A_{i},M_{j}) \delta_{a_{i}, a_{j}} \\
& = \mathbf{K}_{1M}^{an\lambda} (A_i,an\lambda\mu) \mathbf{b}_{M}^{an\lambda} \\
& \approx [\mathbf{K}_{1M}^{an\lambda} (A_i,an\lambda\mu) \mathbf{V}_{MD}^{an\lambda} (\mathbf{\Lambda}_{DD}^{an\lambda})^{-1/2}]
[(\mathbf{\Lambda}_{DD}^{an\lambda})^{1/2} (\mathbf{V}_{MD}^{an\lambda})^{\top} \mathbf{b}_{M}^{an\lambda}] \\
% & = \sum\limits_{d}^{D^{an\lambda}} \tilde{b}_{d}^{an\lambda} \psi_{d}^{an\lambda} (A_{i}; an\lambda\mu) \\
& \equiv \mathbf{\Psi}_{1D} (A_{i}; an\lambda\mu) \tilde{\mathbf{b}}_{D}^{an\lambda}
\end{aligned}
$$



### GPR optimization

#### Direct inversion

In the [initial SALTED paper](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00576  ), optimized regression weights $\mathbf{b}$ are obtained by direct inversion (without RKHS reformulation)

$$
\mathbf{b}_{M} = \left(\mathbf{K}_{NM}^{\top} \mathbf{S}_{NN} \mathbf{K}_{NM}
+ \eta \mathbf{K}_{MM} \right)^{-1} \mathbf{K}_{NM}^{\top} \mathbf{w}_{N}
$$

This method is no longer suggested in SALTED, since it leads to unavoidable instabilities.


#### Conjugate gradient method

With the RKHS reformulation, the regression loss function is a quadratic function of the GPR weights,
and we can apply the conjugate gradient (CG) method to solve the optimization problem.
This is discussed in [this paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00850  ),
and details of the CG algorithm can be found at [Wikipedia](https://en.wikipedia.org/wiki/Conjugate_gradient_method).

The loss function we minimize by CG in this case is

$$
l(\tilde{\mathbf{b}}_{D}) = (\mathbf{\Psi}_{ND} \tilde{\mathbf{b}}_D - \textbf{c}_N^{\text{df}})^{T}\mathbf{S}_{NN}(\mathbf{\Psi}_{ND} \tilde{\mathbf{b}}_D - \textbf{c}_N^{\text{df}}) + \eta \tilde{\mathbf{b}}_{D}^{T} \tilde{\mathbf{b}}_{D}
$$

where $\textbf{c}_N^{\text{df}}$  are the known fitted RI/density-fitting coefficients. Note the need of the overlap matrix $\mathbf{S}_{NN}$ in each case -- a consequence of a non-orthogonal basis --  which will be discussed further on.


