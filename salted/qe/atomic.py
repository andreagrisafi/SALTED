from unittest import result

import numpy as np
import scipy
from sympy.physics.quantum.cg import CG
from collections.abc import Callable
import scipy.special as sp
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
import linecache

from .core import RadialFunctions

def Clebsch_Gordan_coeff(l1: int, m1: int, l2: int, m2: int) -> dict:
    """
    Prepares Clebsch Gordan coefficients for multiplication of spherical harmonics.

    Parameters
    ----------
    l1 : int
        l quantum number of the first spherical harmonic.
    m1 : int
        m quantum number of the first spherical harmonic.
    l2 : int
        l quantum number of the second spherical harmonic.
    m2 : int
        m quantum number of the second spherical harmonic.

    Returns
    -------
    dict
        A dictionary with the Clebsch Gordan coefficients for the multiplication
        of the two spherical harmonics. Keys are the resulting (L, M) values
        of the product spherical harmonic, and values are the corresponding
        Clebsch Gordan coefficients.
    """
    coeffs = {}
    for L in range(np.abs(l1 - l2), l1 + l2 + 1):
        print(L)
        M = m1 + m2
        coeff = np.sqrt((2 * l1 + 1) * (2 * l2 + 1)/ (4 * np.pi * (2 * L + 1))) * CG(l1, 0, l2, 0, L, 0) * CG(l1, m1, l2, m2, L, M)
        coeffs[(L, M)] = coeff.doit() #convert from sympy expression to a numerical value

    
    return coeffs


def multiply_spherical_harmonics(l1: int, m1: int, l2: int, m2: int) -> Callable:
    """
    Multiplies two spherical harmonics and returns the resulting function.

    Parameters
    ----------
    l1 : int
        l quantum number of the first spherical harmonic.
    m1 : int
        m quantum number of the first spherical harmonic.
    l2 : int
        l quantum number of the second spherical harmonic.
    m2 : int
        m quantum number of the second spherical harmonic.

    Returns
    -------
    Callable
        A function that takes theta and phi as arguments and returns the value
        of the product of the two spherical harmonics at those angles.
    """
    coeffs = Clebsch_Gordan_coeff(l1, m1, l2, m2)
    
    def product(theta, phi):
        result = 0
        for (L, M), coeff in coeffs.items():
            Y_LM = sp.sph_harm_y(L, M, theta, phi)
            result += coeff * Y_LM
        return result
    
    return product


class Atomic_Rad(RadialFunctions):
    """Set of atomic radial functions derived from multiplying atomic wavefunctions.

    This class reads radial functions from a file, multiplies them to form a product
    basis, and then compresses the basis by diagonalizing the overlap matrix.

    Parameters
    ----------
    species : str
        The atomic species of this set of radials.
    origin : tuple[float, float, float]
        Origin of all radial functions.
    n_max : int | list[int]
        Highest major quantum number contained in the basis or list of n_max for each l.
    l_max : int
        Highest minor quantum number contained in the basis.
    filename : str
        The path to the file containing the radial functions to be read.
    """

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int | list[int], l_max: int, filename: str):
        super().__init__(species, origin, n_max, l_max)
        l_list, rfunc_list = self._read_from_file(filename)
        wfcs_sq, lprime_list = self._multiply_radial_fcts(l_list, rfunc_list, lprime_max=self.l_max)
        self.radials, self.l_list = self._compress_basis(wfcs_sq, lprime_list, overlap_ninds=self.n_max)
        self.cutoff = self.estimate_cutoff(threshold=1e-5)

    def estimate_cutoff(self, threshold: float = 1e-5) -> float:
        """
        Estimates the cutoff radius where all radial functions decay below a threshold.

        Parameters
        ----------
        threshold : float
            The threshold value relative to the maximum amplitude. Default is 1e-5.

        Returns
        -------
        float
            The estimated cutoff radius.
        """
        r = np.linspace(0, 20.0, 1000)
        radial_values = [radial_func(r) for radial_func in self.radials]
        max_value = np.max(np.abs(radial_values))
        cutoff_indices = np.where(np.abs(radial_values) < threshold * max_value)[0]
        if len(cutoff_indices) == 0:
            return r[-1]
        else:
            return r[cutoff_indices[0]]

    def _compute_norm_amplitude(self, method: str = "analytical") -> float:
        match method:
            case "analytical":
                raise NotImplementedError("Analytical normalization not implemented for this type of radial function.")
            case "numerical":
                def integrand(r):
                    return r**2 * self.__call__(r)**2
                integral, _ = quad(integrand, 0, np.inf)
                amplitude = np.sqrt(1.0 / integral)
            case _:
                raise ValueError(f"Unknown normalization method: {method}")
        return amplitude

    @staticmethod
    def _compute_radial_overlap(r1: Callable, r2: Callable, r_max: float, n_points: int) -> float:
        """Computes the overlap integral between two radial functions."""
        r = np.linspace(0, r_max, n_points)
        integrand = r1(r) * r2(r) * r**2
        return np.trapezoid(integrand, r)

    @staticmethod
    def _select_l1l2(l_prime: int) -> list[tuple[int, int]]:
        """Selects allowed pairs of l1 and l2 based on the triangle inequality."""
        allowed_pairs = []
        for l1 in range(0, l_prime + 1):
            for l2 in range(l1, l_prime + 1):
                if np.abs(l1 - l2) <= l_prime <= (l1 + l2):
                    allowed_pairs.append((l1, l2))
        return allowed_pairs

    def _multiply_radial_fcts(self, l_list: np.ndarray, rfunc_list: list[Callable], lprime_max: int) -> tuple[np.ndarray, list[int]]:
        """Multiplies radial functions corresponding to different l values."""
        rfunc_products = []
        lprime_list = []
        for l_prime in range(lprime_max + 1):
            allowed_pairs = self._select_l1l2(l_prime)

            for (l1, l2) in allowed_pairs:
                l1ind = np.where(l_list == l1)[0]
                l2ind = np.where(l_list == l2)[0]
                for ind in l1ind:
                    for ind2 in l2ind:
                        f1 = rfunc_list[ind]
                        f2 = rfunc_list[ind2]

                        def product_rfunc(r, f1=f1, f2=f2):
                            return f1(r) * f2(r)
                        rfunc_products.append(product_rfunc)
                        lprime_list.append(l_prime)

        return np.array(rfunc_products), lprime_list

    @staticmethod
    def _make_rfunc(vec, funcs):
        """Creates a linear combination of radial functions based on a coefficient vector."""
        def new_rfunc(r):
            return sum(v * f(r) for v, f in zip(vec, funcs))
        return new_rfunc

    def _compress_basis(self, radial_basis: np.ndarray, lprime_list: list[int], overlap_ninds: list[int], rmax: float = 10.0, n_points: int = 1000) -> tuple[list[Callable], list[int]]:
        """Compresses the product basis by keeping eigenvectors of the overlap matrix."""
        compressed_basis = []
        lprime_list_new = []
        lprime_values = set(lprime_list)
        for l_prime in lprime_values:
            rfuncs_for_lprime = radial_basis[np.array(lprime_list) == l_prime]
            n_funcs = len(rfuncs_for_lprime)
            overlap_matrix = np.zeros((n_funcs, n_funcs))
            for i in range(n_funcs):
                for j in range(n_funcs):
                    overlap_matrix[i, j] = self._compute_radial_overlap(rfuncs_for_lprime[i], rfuncs_for_lprime[j], r_max=rmax, n_points=n_points)

            eigenvalues, eigenvectors = np.linalg.eigh(overlap_matrix)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            for i in range(overlap_ninds[l_prime]):
                vec = sorted_eigenvectors[:, i]
                new_rfunc = self._make_rfunc(vec, rfuncs_for_lprime)
                compressed_basis.append(new_rfunc)
                lprime_list_new.append(l_prime)

        return compressed_basis, lprime_list_new

    @staticmethod
    def _read_from_file(filename: str) -> tuple[np.ndarray, list[Callable]]:
        """Reads radial functions from a file."""
        data = np.genfromtxt(filename, dtype = np.float64, skip_header=2)
        R_grid = data[:,1]
        radial_wfcs = data[:,2:]
        radial_wfcs = (radial_wfcs.T/R_grid).T
        l_vals = linecache.getline(filename, 2)
        l_vals = np.array(list(map(int, l_vals.split())))
        rfuncs = [
            make_interp_spline(R_grid, radial_wfcs[:, i], k=3)
            for i in range(radial_wfcs.shape[1])
        ]
        return l_vals, rfuncs
