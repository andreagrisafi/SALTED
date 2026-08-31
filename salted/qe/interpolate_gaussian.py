import numpy as np
from collections.abc import Callable
from scipy.optimize import least_squares

from .core import RIBasis, RIBasisSet, RadialFunctions
from .combine_gaussian import CompoundGaussianRadials

def fit_gaussians(r, rho, l, n_gauss, alpha_init=None, alpha_bounds=(1e-4, 1e6)):
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
    alpha_bounds : tuple[float, float]
        The lower and upper bounds for the Gaussian exponents.

    Returns
    -------
    alphas : np.ndarray
        The fitted Gaussian exponents.
    coeffs : np.ndarray
        The fitted coefficients.
    """
    r, rho = np.asarray(r, float), np.asarray(rho, float)
    rl = r**l

    def design(alphas):
        return rl[:, None] * np.exp(-np.outer(r**2, alphas))

    def solve_coeffs(alphas):
        Phi = design(alphas)
        c, *_ = np.linalg.lstsq(Phi, rho, rcond=None)
        return c, Phi

    def residuals(log_alpha):
        c, Phi = solve_coeffs(np.exp(log_alpha))
        return Phi @ c - rho

    if alpha_init is None:
        rmin, rmax = max(r[r > 0].min(), 1e-3), r.max()
        alpha_init = np.geomspace(1/(2*rmax**2), 1/(2*rmin**2), n_gauss)

    log_lo, log_hi = np.log(alpha_bounds[0]), np.log(alpha_bounds[1])
    res = least_squares(residuals, np.log(alpha_init), bounds=(log_lo, log_hi))

    alphas = np.exp(res.x)
    coeffs, Phi = solve_coeffs(alphas)
    order = np.argsort(alphas)
    alphas, coeffs = alphas[order], coeffs[order]

    return alphas, coeffs


def interpolate_atomic_basis(ri_basis: RIBasis, species: str, position: tuple[float, float, float], num_max: int):

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

    Returns
    -------
    alphas: dict
        A dictionary mapping (n, l) tuples to arrays of alpha parameters for the fitted Gaussian functions.
    coeffs: dict
        A dictionary mapping (n, l) tuples to arrays of coefficients for the fitted Gaussian functions.
    """
    alphas = {}
    coeffs = {}
    #species_basis = ri_basis._load_ri_basis(species, position)
    for idx in range(len(ri_basis.radial_funcs)):
        n, l = ri_basis.radial_funcs.running_to_lexographic_index(idx)
        func = ri_basis.radial_funcs.radials[idx]
        rcut = 10.0  # This can be adjusted based on the specific radial function
        r = np.linspace(0, rcut, 512)
        rho = func(r)
        fitted_alphas, fitted_coeffs = fit_gaussians(r, rho, l, num_max)
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

    