import numpy as np
from collections.abc import Callable
from scipy.optimize import least_squares

from .core import RIBasis, RIBasisSet, RadialFunctions
from .combine_gaussian import CompoundGaussianRadials

def fit_gaussians(r, rho, l, n_gauss, alpha_init=None, alpha_bounds=None, alpha_max=1e6):
    """Fit rho(r) ~ sum_i c_i * r^l * exp(-alpha_i * r^2) via variable projection.

    Parameters
    ----------
    r : np.ndarray
        The radial coordinates.
    rho : np.ndarray
        The target function values.
    l : int
        The angular momentum quantum number.
    n_gauss : int
        The number of Gaussian functions to use.
    alpha_init : np.ndarray, optional
        The initial guess for the Gaussian exponents.
    alpha_bounds : tuple[float, float], optional
        The lower and upper bounds for the Gaussian exponents. If not given, the lower bound is
        tied to the fit domain itself: 1/(2*r.max()**2), i.e. the exponent of a Gaussian whose
        decay length equals r.max() (rcut). Without this, the nonlinear fit is free to drive an
        exponent arbitrarily close to zero to chase a long tail in rho, producing a Gaussian so
        diffuse that a periodic density-fitting real-space lattice sum built from it (see
        salted.cp2k.utils.pair_cutoffs/lattice_images) explodes combinatorially. The upper bound
        defaults to alpha_max (see below) for the mirror-image reason: an unconstrained fit is
        free to chase a sharp near-origin feature with an arbitrarily large exponent, producing a
        Gaussian so spatially sharp that a finite real-space grid (see alpha_max) can't resolve its
        reciprocal-space content -- self-consistent within that grid's own (reciprocal-space)
        density-fitting linear algebra, but wildly wrong when reconstructed from the exact,
        untruncated real-space formula elsewhere (e.g. salted.qe.df2cube).
    alpha_max : float, optional
        Upper bound on the Gaussian exponent when alpha_bounds is not given explicitly. Should be
        tied to the actual real-space grid resolution the fitted basis will later be used on (see
        salted.qe.get_basis_info.max_safe_alpha), not left at its permissive default of 1e6.

    Returns
    -------
    alphas : np.ndarray
        The fitted Gaussian exponents.
    coeffs : np.ndarray
        The fitted coefficients.
    """
    r, rho = np.asarray(r, float), np.asarray(rho, float)
    rl = r**l
    rmin, rmax = max(r[r > 0].min(), 1e-3), r.max()

    if alpha_bounds is None:
        alpha_bounds = (1.0 / (2.0 * rmax**2), alpha_max)
        if alpha_bounds[0] >= alpha_bounds[1]:
            raise ValueError(
                f"fit_gaussians: lower exponent bound {alpha_bounds[0]:.3g} (set by the fit "
                f"domain's own rmax={rmax:.3g}) is not below alpha_max={alpha_max:.3g} (set by the "
                f"real-space grid resolution) -- no exponent can simultaneously decay within rcut "
                f"and stay resolvable on this grid. The atomic cutoff radius is too large for the "
                f"grid this basis will be fit on; shrink it or coarsen alpha_max's threshold."
            )

    def design(alphas):
        return rl[:, None] * np.exp(-np.outer(r**2, alphas))

    def solve_coeffs(alphas):
        Phi = design(alphas)
        # rcond=None (numpy's machine-precision default, ~1e-13 relative here) essentially never
        # truncates: when the nonlinear exponent search (below) lets two alphas drift close
        # together, their Phi columns become nearly collinear and lstsq's minimum-residual solution
        # swings to huge, cancelling coefficients (e.g. +-1e6) to exploit that sliver of remaining
        # distinguishing power. That's a mathematically valid fit (the huge terms cancel to
        # reproduce the right shape) but numerically explosive: this contracted Gaussian's own
        # self-overlap is then computed downstream as a difference of near-equal huge numbers
        # (e.g. in PySCF's contraction normalization), which catastrophically cancels to ~0 or
        # slightly negative and produces NaN/inf from there on. An explicit, much stricter rcond
        # truncates near-collinear directions instead of inverting them, giving the minimum-norm
        # solution on the well-conditioned subspace -- negligible cost in fit quality (a truncated
        # direction contributes almost nothing to the residual by construction), and it removes any
        # incentive for the optimizer to let exponents cluster in the first place.
        c, *_ = np.linalg.lstsq(Phi, rho, rcond=1e-6)
        return c, Phi

    def residuals(log_alpha):
        c, Phi = solve_coeffs(np.exp(log_alpha))
        return Phi @ c - rho

    if alpha_init is None:
        # Clamp the initial guess's own range to alpha_bounds: a tight alpha_max (see above) can
        # be well below the "matches the finest grid spacing" scale 1/(2*rmin**2) this would
        # otherwise reach for, and least_squares requires x0 to lie within bounds.
        init_lo = max(1.0 / (2.0 * rmax**2), alpha_bounds[0])
        init_hi = min(1.0 / (2.0 * rmin**2), alpha_bounds[1])
        alpha_init = np.geomspace(init_lo, init_hi, n_gauss)

    log_lo, log_hi = np.log(alpha_bounds[0]), np.log(alpha_bounds[1])
    res = least_squares(residuals, np.log(alpha_init), bounds=(log_lo, log_hi))

    alphas = np.exp(res.x)
    coeffs, Phi = solve_coeffs(alphas)
    order = np.argsort(alphas)
    alphas, coeffs = alphas[order], coeffs[order]

    return alphas, coeffs


def interpolate_atomic_basis(ri_basis: RIBasis, species: str, position: tuple[float, float, float], num_max: int, alpha_max: float = 1e6):

    """
    Interpolates a set of atomic radial functions using Gaussian functions.

    Parameters
    ----------
    ri_basis : RIBasisSet
        The basis set containing the atomic radial functions to be interpolated.
    species : str
        The chemical symbol of the species.
    position : tuple[float, float, float]
        The position of the atom.
    num_max : int
        The maximum number of Gaussian functions to use for the fit.
    alpha_max : float, optional
        Upper bound on fitted Gaussian exponents, passed through to fit_gaussians (see there). Tie
        this to the real-space grid the basis will actually be density-fit on (e.g. via
        salted.qe.get_basis_info.max_safe_alpha) -- otherwise the fit is free to chase sharp
        near-origin features with exponents the grid can't resolve, which is self-consistent within
        density_fitting.py's own (reciprocal-space, grid-limited) linear algebra but wildly wrong
        once reconstructed from the exact real-space formula (e.g. in df2cube.py).

    Returns
    -------
    alphas: dict
        A dictionary mapping (n, l) tuples to arrays of alpha parameters for the fitted Gaussian functions.
    coeffs: dict
        A dictionary mapping (n, l) tuples to arrays of coefficients for the fitted Gaussian functions.
    """
    alphas = {}
    coeffs = {}
    # Use the radial function set's own data-driven cutoff when it has one (e.g. Atomic_Rad computes
    # this from where its compressed basis actually decays, via estimate_cutoff()), so the Gaussian
    # fit domain -- and therefore fit_gaussians' default lower exponent bound -- reflects the real
    # support of this species' atomic reference data rather than an arbitrary fixed radius.
    rcut = getattr(ri_basis.radial_funcs, "cutoff", None) or 10.0
    #species_basis = ri_basis._load_ri_basis(species, position)
    for idx in range(len(ri_basis.radial_funcs)):
        n, l = ri_basis.radial_funcs.running_to_lexographic_index(idx)
        func = ri_basis.radial_funcs.radials[idx]
        r = np.linspace(0, rcut, 512)
        rho = func(r)
        fitted_alphas, fitted_coeffs = fit_gaussians(r, rho, l, num_max, alpha_max=alpha_max)
        alphas[(n, l)] = fitted_alphas
        coeffs[(n, l)] = fitted_coeffs

    rad_basis_new = CompoundGaussianRadials(species, position, ri_basis.radial_funcs.n_max, ri_basis.radial_funcs.l_max, alphas=alphas, coeffs=coeffs)

    return rad_basis_new


def interpolate_basis_set(ri_basis: RIBasisSet, filename: str):
    """
    Writes the basis set to a file in a format compatible with Quantum ESPRESSO.

    Parameters
    ----------
    ri_basis : RIBasisSet
        The basis set to be written to the file.
    filename : str
        The name of the file to write the basis set to.
    """
    for species, position in ri_basis.species_and_positions:
        ribasis = ri_basis.get_ribasis(species)
        ri_basis = interpolate_atomic_basis(ribasis, species, position, num_max=5)  # Example: using 5 Gaussian functions for interpolation

def print_basis_set(ri_basis: RIBasisSet, filename: str):
    """
    Prints the basis set in a format compatible with SALTED

    Parameters
    ----------
    ri_basis : RIBasisSet
        The basis set to be printed.
    filename : str
        The name of the file to write the basis set to.
    """
    with open(filename, "w") as f:
        for species, position in ri_basis.species_and_positions:
            ribasis = ri_basis.load_ribasis(species)
            f.write(f"Species: {species}, Position: {position}\n")
            for idx in range(len(ribasis)):
                n, l = ribasis.running_to_lexographic_index(idx)
                func = ribasis.radials[idx]
                f.write(f"n={n}, l={l}, Function: {func}\n")

    