from typing import Dict, Tuple
from .core import RadialFunctions
import numpy as np
from scipy.integrate import quad
from scipy.special import gamma


class PrimitiveGaussianRadials(RadialFunctions):
    """Set of primitive Gaussian Radial functions, used as the RadialFunctions class when constructing an RI-Basis.

    The class represents a collection of primitive Gaussian radial functions, that is:
        {R_{nl}(r) = A(n,l) * r^l * exp(-alpha(n,l) * r^2)}

    The set {R_{nl}(r)} is associated with a position, R, and an atomic species.

    The size of the set is defined by the maximum major quantum number, n_max, and maximum minor quantum number, l_max. To define the Gaussian
    radials, each combination of (n, l) requires an alpha-value, which needs to be provided as a key-word argument.

    Parameters
    ----------
        species (str): The atomic species of this set of radials
        origin (tuple[float, float, float]): Origin of all radial functions
        n_max (int): Highest major quantum number contained in the basis
        l_max (int): Highest minor quantum number contained in the basis

        alphas(Dict | Tuple): Alpha values, can be passed as dictionary or tuple.
            Dict: If passed as dictionary, needs to be passed as a mapping of (n,l) tuples to alpha values, e.g.
            {(1, 0): 1.0, (1, 1): 0.5, (2, 0): 0.3 ...)
            Tuple: If passed as tuple, alpha values are expected to be in lexographic order (n,l), where n runs slow and l fast, i.e.
            (alpha(1,0), alpha(1,1), alpha(2,0), alpha(2,1), ...)    
    """

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int | list[int], l_max: int,
                 alphas: list[float] | Dict[Tuple[int, int], float]):
        super().__init__(species, origin, n_max, l_max)

        if isinstance(alphas, list):
            if len(alphas) != len(self):
                raise ValueError(f"Expected {len(self)} alphas for n_max={self.n_max} and l_max={self.l_max}, but got {len(alphas)}.")
            self.alphas = {self.running_to_lexographic_index(idx): alpha for idx, alpha in enumerate(alphas)}
        elif isinstance(alphas, dict):
            expected_keys = {self.running_to_lexographic_index(idx) for idx in range(len(self))}
            if set(alphas.keys()) != expected_keys:
                raise ValueError(f"Expected keys {expected_keys} for alphas dict, but got {set(alphas.keys())}.")
            self.alphas = alphas
        else:
            raise TypeError("alphas must be provided as a list or dict keyed by (n, l).")

        self.radials = []
        for idx in range(len(self)):
            n, l = self.running_to_lexographic_index(idx)
            alpha = self.alphas[(n, l)]
            radial_func = PrimitiveGaussian(alpha, l)
            self.radials.append(radial_func)


    def estimate_cutoff(self, threshold: float = 1e-6) -> float:
        min_alpha = min(self.alphas.values())
        return np.sqrt(-np.log(threshold) / min_alpha)



class PrimitiveGaussian:
    """Primitive Gaussian radial function.

    Represents a radial function of the form

        R(r) = A * r**l * exp(-alpha * r**2),

    where ``alpha`` is the Gaussian exponent, ``l`` is the angular momentum,
    and ``A`` is a normalization constant chosen such that

        integral_0^infinity |R(r)|^2 r^2 dr = 1.

    Parameters
    ----------
    alpha : float
        Gaussian exponent controlling the radial decay.
    l : int
        Angular momentum quantum number. Determines the polynomial prefactor
        ``r**l``.
    """
    def __init__(self, alpha: float, l: int):
        self.alpha = alpha
        self.l = l
        self.amplitude = self._compute_norm_amplitude("analytical")


    def __call__(self, r: float | np.ndarray) -> float | np.ndarray:
        r_arr = np.asarray(r)

        result = self.amplitude * np.exp(-self.alpha * r_arr**2) * r_arr**self.l

        return result

    def _compute_norm_amplitude(self, method: str = "analytical") -> float:
        match method:
            case "analytical":
                k = (3.0 + 2.0 * self.l) / 2.0
                I = 0.5 * (2.0 * self.alpha) ** (-k) * gamma(k)
                amplitude = float(np.sqrt(1.0 / I))
            case "numerical":
                def integrand(r):
                    return r**2 * self.__call__(r)**2
                integral, _ = quad(integrand, 0, np.inf)
                amplitude = np.sqrt(1.0 / integral)
            case _:
                raise ValueError(f"Unknown normalization method: {method}")
        return amplitude




