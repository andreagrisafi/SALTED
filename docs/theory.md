# Theory 

### Density fitting method

The representation of the electron density follows the density fitting (DF), or resolution of the identity (RI), method commonly used in quantum chemistry. This implies approximating the self-consistent electron density $n_{e}$, as computed by the selected electronic-strucutre code, through a linear expansion over atom-centered radial functions $R_{n\lambda}$ and spherical harmonics $Y_{\mu}^{\lambda}$:

$$
n_{e}(\boldsymbol{r}) \approx \sum_{in\lambda\mu} c_{i}^{n\lambda\mu} \sum_{\boldsymbol{u}} \phi_{n\lambda\mu}\left(\boldsymbol{r}-\boldsymbol{r_{i}} -\boldsymbol{u}\right)  
=  \sum_{in\lambda\mu} c_{i}^{n\lambda\mu} \sum_{\boldsymbol{u}} R_{n\lambda}(\left|\boldsymbol{r}-\boldsymbol{r_{i}} -\boldsymbol{u}\right|)Y_{\mu}^{\lambda}(\widehat{\boldsymbol{r}-\boldsymbol{r_{i}}-\boldsymbol{u}}) 
$$

where $\phi$ is a compact symbol for the auxiliary functions of indexes $n\lambda\mu$, $i$ indicates the atomic index in the unit cell, $\boldsymbol{u}$ is the cell translation vector (assuming the system is periodic), and $c_{i}^{nlm}$ are the density-fitting expansion coefficients. 

Different metrics can be chosen to perform the density fitting from the reference self-consistent density, e.g., overlap, Coulomb, ..., depending on the target application, as well as on the options provided by the selected electronic-structure code. This metric will be similarly used in the SALTED loss function. In fact, because of the non-orthogonal nature of the basis functions, the 2-center auxiliary integrals of the form $\bra{\phi}\hat{O}\ket{\phi'}$ are needed to couple the expansion coefficients together. When the overlap metric is adopted, the operator $\hat{O}$ is defined to be the identity. Conversely, the integral operator $\hat{O} = \int d\boldsymbol{r'} 1/|\boldsymbol{r}-\boldsymbol{r'}|$ is applied when using the Coulomb metric. Note that, in practice, the Coulomb potential must be truncated when considering a periodic system, so that a truncated Coulomb metric is typically adopted in this case.  

### Symmetry-adapted descriptor

SALTED uses symmetry-adapted Gaussian process regression (SAGPR) to predict the density coefficients. The calculation of covariant kernel functions for each angular momentum $\lambda$ used to expand the electron density follows what presented in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002). In particular, these are defined as inner products between three-body spherical equivariants of order $\lambda\mu$ built from a given representation $X$ of the local environmnet of atom $i$. In abstract Dirac notation, this is given by

$$
\ket{P_{i}^{\lambda\mu}} = \int d\hat{R} \left(\hat{R}\ket{X_{i}} \otimes \hat{R}\ket{X_{i}'} \otimes \hat{R}\ket{\lambda\mu}\right)
$$

In SALTED, $X$ and $X'$ can independently be chosen as density-like or potential-like representations. When the former choice is adopted for both representations, the descriptor reduces to the $\lambda$-SOAP power spectrum, as introduced in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002). When the latter choice is made for at least one of the two representations, the model will possess long-range information following the LODE method as presented in [Chem. Sci. 12, 2078-2090 (2021)](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc04934d). In practice, the expansion coefficients $X_{i}^{anlm}$ of both density-like and potential-like representations are computed for each atomic type $a$ on a basis of orthogonal radial functions $n$ and spherical harmonics $lm$ using the [rascaline package](https://github.com/Luthaf/rascaline), and used in SALTED to compute the three-body spherical equivariants following the prescription reported in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002), i.e.,

$$
 P_{i}^{\lambda \mu}(aa'nn'll') = (-1)^{\lambda-l'} \sum_{m}\  X_{i}^{anlm}\ \left(X_{i}^{a'n'l'(m-\mu)}\right)^\star (-1)^m \begin{pmatrix}
   \lambda & l' & l \\
   \mu & (m-\mu) & -m 
  \end{pmatrix}
$$

where the set of indexes $aa'nn'll'$ identify the features space.
Since the electron density is expanded on a basis of real spherical harmonics, the complex to real transformation matrix:

$$
\mathbf{C}^\lambda = \frac{1}{\sqrt{2}}\left( 
\begin{matrix}
i & 0 & ...& 0 & ... & 0 & -i(-1)^\lambda\\
0 & i & ... & 0 & ... & -i(-1)^{\lambda-1} & 0 \\
... & ... & ... & ...& ...& ...& ...\\
0 & 0 & .. & \sqrt{2} & .. & 0 & 0\\
&... &... &... &... &... &...\\
0 & 1 & ... & 0 & ... & (-1)^{\lambda-1} & 0\\
1 & 0 & ... & 0 & ... & 0 & (-1)^\lambda 
\end{matrix},
\right)
$$

is applied to the equivariant descriptor: 


$$
\boldsymbol{\mathcal{P}}^{\lambda} = \mathbf{C}^\lambda \cdot \boldsymbol{P}^{\lambda}
$$ 

Importantly, the descriptor so computed will still have an imaginary part. At this point, we can enforce inversion symmetry to make the descriptor equivariant in O(3) by only retaining the components for which $l+l'+\lambda$ is even. Upon the complex to real transformation previously performed, this implies that we can in fact discard the imaginary part of $\boldsymbol{\mathcal{P}}^{\lambda}$, obtaining 

$$
\boldsymbol{\mathcal{P}}^{\lambda,O(3)} = Real[\boldsymbol{\mathcal{P}}^{\lambda}] 
$$

Finally, we apply the following normalization

$$
\boldsymbol{\tilde{\mathcal{P}}}^{\lambda,O(3)} = \boldsymbol{\mathcal{P}}^{\lambda,O(3)}/\sqrt{\boldsymbol{\mathcal{P}}^{\lambda,O(3)}\cdot \left(\boldsymbol{\mathcal{P}}^{\lambda,O(3)}\right)^T} 
$$


### Symmetry-adapted kernel

From the normalized symmetry-adapted descriptors $\boldsymbol{\tilde{\mathcal{P}}}^{\lambda,O(3)}$, symmetry-adapted kernel functions that couple two given atomic environments $i$ and $j$ are computed as follows:

$$
k_{\mu\mu'}^{\lambda}(i,j) = \boldsymbol{\tilde{\mathcal{P}}}^{\lambda\mu,O(3)}(i) \cdot \left(\boldsymbol{\tilde{\mathcal{P}}}^{\lambda\mu',O(3)}(j)\right)^T     
$$

Non-linear kernels can then be constructed by moltiplying them by their scalar ($\lambda=0$) counterpart, elavated to a positive integer $z>0$:

$$
\tilde{\boldsymbol{k}}^{\lambda}(i,j) = \boldsymbol{k}^{\lambda}(i,j) \times \left(k^{0}(i,j)\right)^{z-1} 
$$


### Subset of regressors approximation

In order to reduce the dimensionality of the regression problem, we adopt the subset of regressors approach (SoR). For each $\lambda$, in particular, we approximate the full kernel matrix between the $\mathcal{N}$ atoms of the training set using a subset of $M<\mathcal{N}$ atomic environments, such that

$$
K_{\mathcal{N}\mathcal{N}} \approx K_{\mathcal{N}M} K_{MM}^{-1} K_{M\mathcal{N}}
$$

In SALTED, $M$ is selected using the Farhest Point Sampling algorithm (see the FPS section). In practice, $K_{MM}$ is most of the time found to be low rank, so that a suitable strategy must be adopt to numerically stabilize the problem (see the RKHS section).

### Farthest point sampling

FPS is a (simple but useful) greedy algorithm to select a diverse subset of points from a given set.
Please check, e.g., this [blog](https://minibatchai.com/2021/08/07/FPS.html) for details. In SALTED, the FPS algorithm is used to (i) sparsify the features space of each spherical equivariant of order $\lambda$, and (ii) to select the most representative atomic environments $M$ based on the scalar kernel metric.

### Symmetry-adapted prediction

Within the sparse-GPR formalism, the density coefficients $c_{i}^{n\lambda\mu}$ of a given atom $i$ are predicted as a linear combination of the kernel functions

$$
c_{i}^{n\lambda\mu} \approx \sum\limits_{j \in M} \sum\limits_{|\mu'| \le \lambda}
b_{n\lambda\mu'}(j) k_{\mu\mu'}^{\lambda}(i,j) \delta_{a_{i}, a_{j}}
$$

where $b_{n\lambda\mu'}(j)$ are the regression weights and j runs over the sparse selection of $M$ atom. 
Note tha a Kronecker delta function $\delta_{a_{i}, a_{j}}$ is introduced to make sure the atomic species of atom $i$ and atom $j$ are the same. 

### Reproducing kernel Hilbert space (RKHS)

In order to numerically stabilize the learning procedure, it is convenient to recast the sparse-GPR problem into the features space that underlies the contruction of the (non-linear) kernel functions. 
Following the [representer theorem](https://en.wikipedia.org/wiki/Representer_theorem), 
it is in fact possible to find a Hilbert space with elements $\mathbf{\Psi}$ whose inner product reproduces the kernels. This is commonly referred to as the reproducing kernel Hilbert space (RKHS).  

For each channel $an\lambda$, we first perform an SVD decomposition of $\left(\mathbf{K}_{MM}^{an\lambda}\right)^{-1}$ by keeping only the non-negligible eigenvalues $\lambda_{d}$ (above a given threshold $\epsilon$) and their respective eigenvectors  $\mathbf{v}_{d}$. Following the SoR approximation, we then write 


$$
\mathbf{K}_{NN}^{an\lambda} \approx \mathbf{K}_{NM}^{an\lambda} \sum_d^{D^{an\lambda}} \frac{\mathbf{v}_d^M (\mathbf{v}_d^M)^T}{\lambda_d}  \mathbf{K}_{MN}^{an\lambda}
$$

where we assumed

$$
\left(\mathbf{K}_{MM}^{an\lambda})^{-1}
\approx \sum\limits_{d}^{D^{an\lambda}} \mathbf{v}_{M}^{d} \lambda_{d}^{-1} (\mathbf{v}_{M}^{d})^{\top}
= \mathbf{V}_{MD}^{an\lambda} (\mathbf{\Lambda}_{DD}^{an\lambda})^{-1} (\mathbf{V}_{MD}^{an\lambda})^{\top}
$$

where $D^{an\lambda}$ is the final truncated dimension, $\mathbf{\Lambda}_{DD}^{an\lambda}$ is the diagonal matrix of selected eigenvalues, and the $\mathbf{V}_{MD}^{an\lambda}$ is the batched selected eigenvectors.
With this expression, we can define the RKHS of $\mathbf{K}_{NN}^{an\lambda}$ as the space spanned by the following feature vectors 

$$
\begin{aligned}
\mathbf{\Psi}_{ND}^{an\lambda} \equiv & \mathbf{K}_{NM}^{an\lambda} \mathbf{V}_{MD}^{an\lambda} (\mathbf{\Lambda}_{DD}^{an\lambda})^{-1/2} \\
\end{aligned}
$$

which yields (as expected)

$$
\mathbf{K}_{NN}^{an\lambda} \approx \mathbf{\Psi}_{ND}^{an\lambda} (\mathbf{\Psi}_{ND}^{an\lambda})^{\top}
$$

With the RKHS reformulation above, we can finally reformulate the density-learning problem as a linear regression task parametrized according to the feature vectors $\mathbf{\Psi}_{ND}^{an\lambda}$.
In particular, the prediction of the density coefficients is rewritten as follows

$$
c_{i}^{n\lambda\mu} 
\approx \mathbf{\Psi}_{D}^{an\lambda\mu}(i) \tilde{\mathbf{b}}_{D}^{an\lambda}
$$

### SALTED loss function and solution 

Within the RKHS reformulation, the SALTED loss function is written as a quadratic form of the regression weights. Assuming an overlap metric, this is given by:  

$$
\mathcal{L}(\tilde{\mathbf{b}}_{D}) = (\mathbf{\Psi}_{ND} \tilde{\mathbf{b}}_D - \textbf{c}_N^{\text{DF}})^{T}\mathbf{S}_{NN}(\mathbf{\Psi}_{ND} \tilde{\mathbf{b}}_D - \textbf{c}_N^{\text{DF}}) + \eta \tilde{\mathbf{b}}_{D}^{T} \tilde{\mathbf{b}}_{D}
$$

where $\textbf{c}_N^{\text{DF}}$ is the vector of reference density-fitting coefficients associated with $N$ training configurations, while $\eta$ is a regularization parameters which acts as a penalty to high-norm weights. 

- Explicit solution

The problem can be analytically solved by explicit differentiation of the loss function with respect to the regression weights, obtaining

$$
\tilde{\mathbf{b}}_D = \left(\mathbf{\Psi}_{ND}^T \cdot \mathbf{S}_{NN} \cdot \mathbf{\Psi}_{ND} + \eta \mathbf{\Psi}_{DD} \right)^{-1} \left(\mathbf{\Psi}_{ND}^T \cdot \textbf{c}_N^{\text{DF}}\right)   
$$

- Conjugate gradients minimization

When the problem dimensionality $D$ is too large, it is more convenient to numerically minimize the loss function directly. In SALTED, we apply the conjugate gradient (CG) method to solve the optimization problem.
This is discussed in [this paper](https://pubs.acs.org/doi/full/10.1021/acs.jctc.2c00850).
