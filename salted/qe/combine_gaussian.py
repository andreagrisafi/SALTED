from typing import Dict, Tuple
from .core import RadialFunctions
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma



class CompoundGaussianRadials(RadialFunctions):
    """Set of compound Gaussian Radial functions, used as the RadialFunctions class when constructing an RI-Basis.

    The class represents a collection of compound Gaussian radial functions, that is:
        {R_{nl}(r) = A(n,l) * r^l * Sum_{i=1}^N c_i(n,l) * exp(-alpha_i(n,l) * r^2)}

    The set {R_{nl}(r)} is associated with a position, R, and an atomic species.

    The size of the set is defined by the maximum major quantum number, n_max, and maximum minor quantum number, l_max. To define the Gaussian
    radials, each combination of (n, l) requires a set of alpha-values and coefficients, which needs to be provided as a key-word argument.

    Parameters
    ----------
        species (str): The atomic species of this set of radials
        origin (tuple[float, float, float]): Origin of all radial functions
        n_max (int): Highest major quantum number contained in the basis
        l_max (int): Highest minor quantum number contained in the basis

        alphas(Dict | Tuple): Alpha values, can be passed as dictionary or tuple.
            Dict: If passed as dictionary, needs to be passed as a mapping of (n,l) tuples to alpha values, e.g.
            {(1, 0): [1.0, 0.5], (1, 1): [0.3], (2, 0): [0.2, 0.1]}
            Tuple: If passed as tuple, alpha values are expected to be in lexographic order (n,l), where n runs slow and l fast, i.e.
            ([alpha(1,0)_1, alpha(1,0)_2], [alpha(1,1)_1], [alpha(2,0)_1, alpha(2,0)_2], ...)"""
    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int | list[int], l_max: int,
                 alphas: list[float] | Dict[Tuple[int, int], float], coeffs: list[float] | Dict[Tuple[int, int], float]):
        super().__init__(species, origin, n_max, l_max)

        if isinstance(alphas, list):
            if len(alphas) != len(self):
                raise ValueError(f"Expected {len(self)} alphas for n_max={self.n_max} and l_max={self.l_max}, but got {len(alphas)}.")
            self.alphas = {self.running_to_lexographic_index(idx): alpha for idx, alpha in enumerate(alphas)}
            self.coeffs = {self.running_to_lexographic_index(idx): coeff for idx, coeff in enumerate(coeffs)}
        elif isinstance(alphas, dict):
            expected_keys = {self.running_to_lexographic_index(idx) for idx in range(len(self))}
            if set(alphas.keys()) != expected_keys:
                raise ValueError(f"Expected keys {expected_keys} for alphas dict, but got {set(alphas.keys())}.")
            if set(coeffs.keys()) != expected_keys:
                raise ValueError(f"Expected keys {expected_keys} for coeffs dict, but got {set(coeffs.keys())}.")
            self.alphas = alphas
            self.coeffs = coeffs
        else:
            raise TypeError("alphas must be provided as a list or dict keyed by (n, l).")

        self.radials = []
        for idx in range(len(self)):
            n, l = self.running_to_lexographic_index(idx)
            alpha_arr = self.alphas[(n, l)]
            coeff_arr = self.coeffs[(n, l)]
            radial_func = CompoundGaussian(alpha_arr, coeff_arr, l, normalized=True)
            self.radials.append(radial_func)


    def estimate_cutoff(self, threshold: float = 1e-6) -> float:
        min_alpha = min(self.alphas.values())
        return np.sqrt(-np.log(threshold) / min_alpha)

class CompoundGaussian:
    """Linear combination of Gaussian radial functions.

    Represents a radial function of the form

        R(r) = A * r**l Sum_{i=1}^N c_i * exp(-alpha_i * r**2),

    where ``alpha_i`` are the Gaussian exponents, ``l`` is the angular momentum,
    and ``A`` is a normalization constant chosen such that

        integral_0^infinity |R(r)|^2 r^2 dr = 1.

    Parameters
    ----------
    alphas : np.ndarray
        Array of Gaussian exponents controlling the radial decay.
    coeffs : np.ndarray
        Array of coefficients for each Gaussian term.
    l : int
        Angular momentum quantum number. Determines the polynomial prefactor
        ``r**l``.
    normalized : bool, optional
        When True, apply the normalization prefactor to the raw Gaussian sum.
    """
    def __init__(self, alphas: np.ndarray, coeffs: np.ndarray, l: int, normalized: bool = True):
        self.alphas = np.asarray(alphas, dtype=float)
        self.coeffs = np.asarray(coeffs, dtype=float)
        self.l = int(l)
        self.normalized = normalized
        self.amplitude = 1.0
        self.amplitude = self._compute_norm_amplitude("numerical")

    def _raw_value(self, r: float | np.ndarray) -> float | np.ndarray:
        r_arr = np.asarray(r)
        return np.sum([self.coeffs[i] * np.exp(-self.alphas[i] * r_arr**2) * r_arr**self.l
                for i in range(len(self.alphas))
            ],
            axis=0,
        )

    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        raw = self._raw_value(r)
        # if self.normalized:
        #     return self.amplitude * raw
        return raw

    def _compute_norm_amplitude(self, method: str = "numerical") -> float:
        match method:
            case "numerical":
                def integrand(r):
                    return r**2 * self._raw_value(r)**2
                integral, _ = quad(integrand, 0, np.inf)
                if integral <= 0.0 or not np.isfinite(integral):
                    return 1.0
                amplitude = np.sqrt(1.0 / integral)
            case _:
                raise ValueError(f"Unknown normalization method: {method}")
        return amplitude