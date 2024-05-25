# Theory 

### Density fitting method

The representation of the electron density follows the density fitting (DF), or resolution of the identity (RI), method commonly used in quantum chemistry. This implies approximating the electron density $n_{e}$ as a linear expansion over atom-centered radial functions $R_{n\lambda}$ and spherical harmonics $Y_{\mu}^{\lambda}$:

$$
n_{e}(\boldsymbol{r}) \approx \sum_{inlm} c_{i}^{nlm} \sum_{\boldsymbol{u}} \phi_{n\lambda\mu}\left(\boldsymbol{r}-\boldsymbol{r_{i}} -\boldsymbol{u}\right)  
=  \sum_{inlm} c_{i}^{nlm} \sum_{\boldsymbol{u}} R_{n\lambda}(\left|\boldsymbol{r}-\boldsymbol{r_{i}} -\boldsymbol{u}\right|)Y_{\mu}^{\lambda}(\widehat{\boldsymbol{r}-\boldsymbol{r_{i}}-\boldsymbol{u}}) 
$$

where $\phi$ are the auxiliary function, $i$ indicates the atomic index, $\boldsymbol{u}$ to the cell translation vector (assuming the system is periodic) and $c_{i}^{nlm}$ are the expansion coefficients. Several metrics can be chosen to perform density fitting, e.g., overlap, Coulomb, ... This metric  will be similarly used in the SALTED loss function. 

Because of the non-orthogonal nature of the basis functions, the 2-center auxiliary integrals of the form $\bra{\phi}\hat{O}\ket{\phi'}$ are needed to train the model, in addition to the expansion coefficients. The operator $\hat{O}$ is defined to be the identity when the overlap metric is adopted, and the Coulomb operator $1/|\boldsymbol{r}-\boldsymbol{r'}|$ when a Coulomb metric is adopted. 

### Symmetry-adapted descriptor

SALTED uses symmetry-adapted Gaussian process regression (SAGPR) to predict the density coefficients. The calculation of covariant kernel functions for each angular momentum $\lambda$ used to expand the electron density follows what presented in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002). In particular, these are defined as inner products between three-body spherical equivariants of order $\lambda\mu$ built from a given representation $X$ of the local environmnet of atom $i$. In abstract Dirac notation, this is given by

$$
\ket{P_{i}^{\lambda\mu}} = \int d\hat{R} \left(\hat{R}\ket{X_{i}} \otimes \hat{R}\ket{X_{i}'} \otimes \hat{R}\ket{\lambda\mu}\right)
$$

In SALTED, $X$ and $X'$ can independently be chosen as density-like or potential-like representations. When the former choice is adopted for both representations, the descriptor reduces to the $\lambda$-SOAP power spectrum, as introduced in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002). When the latter choice is made for at least one of the two representations, the model will possess long-range information following the LODE method as presented in [Chem. Sci. 12, 2078-2090 (2021)](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d0sc04934d). In practice, the expansion coefficients $X_{i}^{anlm}$ of both density-like and potential-like representations are computed for each atomic type $a$ on a basis of orthogonal radial functions spherical harmonics using the [rascaline package](https://github.com/Luthaf/rascaline), and used in SALTED to compute the three-body spherical equivariants following the prescription reported in [PRL 120, 036002 (2018)](https://link.aps.org/doi/10.1103/PhysRevLett.120.036002), i.e.,

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


### Symmetry-adapted prediction

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


