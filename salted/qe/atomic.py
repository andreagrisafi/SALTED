from unittest import result

import numpy as np
import scipy
import scipy.linalg
from sympy.physics.quantum.cg import CG
from collections.abc import Callable
import scipy.special as sp
from scipy.integrate import quad
from scipy.interpolate import make_interp_spline
import linecache

from .core import RadialFunctions, enumerate_nl_pairs, normalize_n_max

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
    l_max : int
        Highest minor quantum number contained in the basis.
    filename : str
        The path to the file containing the radial functions to be read.
    n_max : int | list[int]
        Number of compressed radial functions to keep per l: for each l, the n_max[l] eigenvectors
        of that l's (generalized, when energy_file is given) overlap-matrix eigenproblem with the
        largest eigenvalue are kept, sorted descending -- i.e. the n_max[l] most independent
        directions. As an int, the same count is used for all l; as a list, one value per l in
        [0, l_max]. There is no automatic eigenvalue-based cutoff underneath this: the atomic
        self-multiplication overlap matrix at a given l only has as many independent
        (well-separated-from-zero) eigenvectors as the input atomic data actually supports, and
        asking for more than that returns near-zero-eigenvalue functions that are numerical noise,
        which can make the downstream density-fitting overlap matrix ill-conditioned or singular,
        or (for a near-null eigenvector) produce NaN coefficients once Gaussian-fitted. Pick n_max
        conservatively per l (e.g. guided by `salted.qe.density_fitting.diagnose_overlap_conditioning`
        on a prior run) rather than relying on this class to reject bad requests for you.
    energy_file : str | None
        Optional path to a `{...}_eigenvalues.dat` file (rows of "l n energy", in Hartree, "#"
        comments ignored) giving each input orbital's energy. When given, compression is biased
        toward low-energy parentage: see `_compress_basis`. When omitted (default), compression is
        purely overlap/completeness-based, as before.
    energy_scale : float
        Energy scale (same units as energy_file, i.e. Hartree) controlling how strongly energy
        competes against overlap/completeness in `_compress_basis`. Only the *relative* energies
        and this scale matter (shifting every energy by a constant has no effect). Smaller values
        bias more strongly toward low-energy shells; larger values approach the pure-overlap
        behavior. Default 1.0 Hartree, comparable to the typical energy gap between zeta shells in
        atomic reference data -- tune per your own data if that's not representative.
    """

    def __init__(self, species: str, origin: tuple[float, float, float], l_max: int, filename: str,
                 n_max: int | list[int],
                 energy_file: str | None = None, energy_scale: float = 1.0):
        l_list, rfunc_list = self._read_from_file(filename)
        e_list = self._read_energies(energy_file, l_list) if energy_file is not None else None
        wfcs_sq, lprime_list, e_prod_list = self._multiply_radial_fcts(l_list, rfunc_list, lprime_max=l_max, e_list=e_list)
        self.n_max_requested = normalize_n_max(n_max, l_max)
        self.energy_scale = energy_scale
        radials, l_list_new, n_max_out = self._compress_basis(
            wfcs_sq, lprime_list, self.n_max_requested, e_prod_list=e_prod_list, energy_scale=energy_scale,
        )
        super().__init__(species, origin, n_max_out, l_max)
        self.radials, self.l_list = radials, l_list_new
        self.cutoff = self.estimate_cutoff(threshold=1e-5)

    @staticmethod
    def _read_energies(filename: str, l_list: np.ndarray) -> np.ndarray:
        """Reads per-orbital energies from a `{...}_eigenvalues.dat` file: rows of "l n energy"
        (comment lines starting with '#' ignored), keyed by (l, n) where n is the zero-indexed
        zeta/shell count for that l. Returns one energy per entry of l_list, in the same order, by
        matching each column of the main radial-function file to its (l, n) -- n being that
        column's zero-indexed position among the columns of l_list sharing the same l, since the
        main file itself carries no explicit n index, only l (see `_read_from_file`). This assumes
        the two files list their orbitals for a given l in the same order.
        """
        energies: dict[tuple[int, int], float] = {}
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                l_str, n_str, e_str = line.split()
                energies[(int(l_str), int(n_str))] = float(e_str)

        n_counters: dict[int, int] = {}
        e_list = np.zeros(len(l_list))
        for i, l in enumerate(l_list):
            n = n_counters.get(int(l), 0)
            n_counters[int(l)] = n + 1
            key = (int(l), n)
            if key not in energies:
                raise ValueError(
                    f"{filename}: missing energy for (l={key[0]}, n={key[1]}), needed for "
                    f"column {i} of the radial-function file."
                )
            e_list[i] = energies[key]
        return e_list

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
        radial_values = np.abs([radial_func(r) for radial_func in self.radials])  # shape (n_funcs, len(r))
        max_value = np.max(radial_values)
        envelope = np.max(radial_values, axis=0)  # per-r max magnitude across all functions
        above_threshold = np.where(envelope >= threshold * max_value)[0]
        if len(above_threshold) == 0:
            return r[-1]
        else:
            return r[above_threshold[-1]]  # largest r at which any function is still significant

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
    def _select_l1l2(l_prime: int, l1l2_max: int) -> list[tuple[int, int]]:
        """Selects allowed pairs (l1, l2), with l1 <= l2 and both in [0, l1l2_max], whose product
        can contain an l_prime component under the angular-momentum triangle inequality
        |l1 - l2| <= l_prime <= l1 + l2.

        l1l2_max is the largest l actually present among the input atomic orbitals -- NOT l_prime
        itself. Two orbitals with large l1, l2 can still couple to a small l_prime (e.g. two p
        orbitals, l1=l2=1, have an l_prime=0 "monopole" component; two d orbitals, l1=l2=2,
        likewise), so capping the search at l1, l2 <= l_prime (as an earlier version of this
        function did) silently drops those pairs and leaves the l_prime=0 (and other low-l_prime)
        product space missing real contributions, regardless of how large l1l2_max is passed in.
        """
        allowed_pairs = []
        for l1 in range(0, l1l2_max + 1):
            for l2 in range(l1, l1l2_max + 1):
                if np.abs(l1 - l2) <= l_prime <= (l1 + l2):
                    allowed_pairs.append((l1, l2))
        return allowed_pairs

    def _multiply_radial_fcts(self, l_list: np.ndarray, rfunc_list: list[Callable], lprime_max: int, e_list: np.ndarray | None = None) -> tuple[np.ndarray, list[int], np.ndarray | None]:
        """Multiplies radial functions corresponding to different l values.

        When e_list is given (one energy per entry of l_list/rfunc_list), also returns the summed
        parent energy E_i + E_j for each product function, in the same order as the returned
        products -- used by _compress_basis to bias compression toward low-energy shells.
        """
        rfunc_products = []
        lprime_list = []
        e_prod_list = [] if e_list is not None else None
        l1l2_max = int(np.max(l_list)) if len(l_list) > 0 else 0
        for l_prime in range(lprime_max + 1):
            allowed_pairs = self._select_l1l2(l_prime, l1l2_max)

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
                        if e_list is not None:
                            e_prod_list.append(e_list[ind] + e_list[ind2])

        e_prod_arr = np.array(e_prod_list) if e_list is not None else None
        return np.array(rfunc_products), lprime_list, e_prod_arr

    @staticmethod
    def _make_rfunc(vec, funcs):
        """Creates a linear combination of radial functions based on a coefficient vector."""
        def new_rfunc(r):
            return sum(v * f(r) for v, f in zip(vec, funcs))
        return new_rfunc

    def _compress_basis(self, radial_basis: np.ndarray, lprime_list: list[int], n_max_by_l: list[int], e_prod_list: np.ndarray | None = None, energy_scale: float = 1.0, rmax: float = 10.0, n_points: int = 1000) -> tuple[list[Callable], list[int], list[int]]:
        """Compresses the product basis by keeping the n_max_by_l[l] eigenvectors of the overlap
        matrix with the largest eigenvalue, for each l (sorted descending, so these are the
        n_max_by_l[l] most independent directions; if n_max_by_l[l] exceeds how many product
        functions exist at that l, all of them are kept instead). There is no eigenvalue-magnitude
        cutoff: the caller is responsible for choosing n_max_by_l small enough that it doesn't
        exceed the data's actual rank at some l (requesting too many returns near-zero-eigenvalue
        noise directions, which can make the downstream density-fitting overlap matrix
        ill-conditioned/singular, or -- for a near-null eigenvector -- yield NaN coefficients once
        Gaussian-fitted; use `salted.qe.density_fitting.diagnose_overlap_conditioning` on a prior
        run to pick n_max_by_l down if this happens).

        When e_prod_list is given (one combined parent-orbital energy per entry of radial_basis,
        from _multiply_radial_fcts), compression solves the generalized eigenproblem
        `overlap_matrix @ v = lambda * W @ v` instead of the plain `overlap_matrix @ v = lambda * v`,
        with `W = diag(exp(E_i / energy_scale))`: a high combined parent energy E_i inflates W_ii,
        shrinking the generalized eigenvalue (effectively "completeness per unit energy cost") for
        directions dominated by that product function. This is deliberately NOT done by just
        rescaling the overlap matrix element-wise (e.g. overlap_matrix[i,j] *= exp(-(E_i+E_j)/2T)):
        scipy's generalized-eigenproblem normalization (v_k^T W v_j = delta_kj) guarantees
        <g_k|g_j> = v_k^T overlap_matrix v_j = lambda_j * delta_kj, i.e. the kept radial functions
        g_k stay mutually orthogonal in the actual physical sense; a plain element-wise reweighting
        would not preserve that and would reintroduce exactly the redundancy this compression
        exists to remove. Shifting every energy by the same constant leaves the eigenvector ranking
        unchanged (it only rescales W by a global factor), so only relative energies and
        energy_scale matter, not an absolute energy zero point. Since this only reorders which
        directions are kept (n_max_by_l is still the hard cap, with no eigenvalue-magnitude cutoff
        underneath), energy weighting cannot by itself introduce singularity/NaN risk beyond what
        plain n_max selection already carries -- but it can trade away some of the raw-overlap
        "most independent" directions for lower-energy ones that are more numerically marginal, so
        it's still worth re-checking cond(S) after turning this on.

        The eigensolve naturally groups functions l-slow/n-fast, but the rest of RadialFunctions
        (running_to_lexographic_index, enumerate_nl_pairs) indexes radials n-slow/l-fast. Build the
        result keyed by (n, l) and re-flatten in that convention so self.radials[idx] stays consistent
        with running_to_lexographic_index(idx).
        """
        radials_by_nl: dict[tuple[int, int], Callable] = {}
        actual_n_max_by_l = [0] * len(n_max_by_l)
        lprime_arr = np.array(lprime_list)
        for l_prime in range(len(n_max_by_l)):
            mask = lprime_arr == l_prime
            rfuncs_for_lprime = radial_basis[mask]
            n_funcs = len(rfuncs_for_lprime)
            if n_funcs == 0:
                continue

            overlap_matrix = np.zeros((n_funcs, n_funcs))
            for i in range(n_funcs):
                for j in range(n_funcs):
                    overlap_matrix[i, j] = self._compute_radial_overlap(rfuncs_for_lprime[i], rfuncs_for_lprime[j], r_max=rmax, n_points=n_points)

            W = np.diag(np.exp(e_prod_list[mask] / energy_scale)) if e_prod_list is not None else None
            eigenvalues, eigenvectors = scipy.linalg.eigh(overlap_matrix, W)
            sorted_indices = np.argsort(eigenvalues)[::-1]
            sorted_eigenvectors = eigenvectors[:, sorted_indices]

            n_keep = min(n_max_by_l[l_prime], n_funcs)
            actual_n_max_by_l[l_prime] = n_keep

            for i in range(n_keep):
                vec = sorted_eigenvectors[:, i]
                radials_by_nl[(i + 1, l_prime)] = self._make_rfunc(vec, rfuncs_for_lprime)

        nl_pairs = enumerate_nl_pairs(actual_n_max_by_l)
        compressed_basis = [radials_by_nl[nl] for nl in nl_pairs]
        lprime_list_new = [l for _, l in nl_pairs]

        return compressed_basis, lprime_list_new, actual_n_max_by_l

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
