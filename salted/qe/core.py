import numpy as np
from ase.io import read
import json

from typing import List, Tuple, Callable, Iterator
from .qe_types import Cutoff, CutoffType


def normalize_n_max(n_max: int | List[int], l_max: int) -> List[int]:
    """Return n_max as a validated list with one entry per l in [0, l_max]."""
    if isinstance(n_max, int):
        if n_max < 0:
            raise ValueError(f"n_max must be non-negative, got {n_max}.")
        return [n_max for _ in range(l_max + 1)]

    n_max_list = [int(value) for value in n_max]
    if len(n_max_list) != l_max + 1:
        raise ValueError(
            f"If n_max is a list, it must have length l_max + 1 ({l_max + 1}), "
            f"but got {len(n_max_list)}."
        )
    if any(value < 0 for value in n_max_list):
        raise ValueError(f"All n_max entries must be non-negative, got {n_max_list}.")
    return n_max_list


def enumerate_nl_pairs(n_max_by_l: List[int]) -> List[Tuple[int, int]]:
    """Enumerate valid (n, l) pairs in lexographic order: n slow, l fast."""
    pairs: List[Tuple[int, int]] = []
    max_n = max(n_max_by_l, default=0)
    for n in range(1, max_n + 1):
        for l, n_l in enumerate(n_max_by_l):
            if n <= n_l:
                pairs.append((n, l))
    return pairs


class RIFunctions:
    """Base class for RI basis functions, providing common attributes and methods for radial, angular and combined functions.
    """

    def __init__(self, species: str, origin: tuple[float, float, float]):
        self.species = species
        self.origin = np.array(origin)

    def compute(self, r: np.ndarray) -> np.ndarray:
        r_arr = np.ascontiguousarray(r)
        if r_arr.ndim != 2 or r_arr.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r_arr.shape}")
        return self.__call__(r_arr - self.origin)


    def __call__(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def __len__(self) -> int:
        raise NotImplementedError


    def lexographic_to_running_index(self, index: tuple) -> int:
        raise NotImplementedError


    def running_to_lexographic_index(self, idx: int) -> tuple:
        raise NotImplementedError


class RadialFunctions(RIFunctions):
    """Base class for all radial functions

    RadialFunctions represents a set of radial functions, {R_{n,l}}, which are identified by their major and minor quantum numbers, n and l.

    An implementation of RadialFunctions has to populate set.radials with a list of callable objects that represent R_{n,l} in lexographic order,
    where n runs slow and l runs fast.

    Calling RadialFunctions with a cartesian point R will give a list of floats, representing the evaulations R_{n,l}(r) in lexographic order.
    """

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int | List[int], l_max: int):
        super().__init__(species, origin)
        self.n_max = normalize_n_max(n_max, l_max)
        #print(self.n_max)
        self.l_max = l_max
        self._nl_pairs = enumerate_nl_pairs(self.n_max)
        self._nl_to_idx = {nl: idx for idx, nl in enumerate(self._nl_pairs)}
        self.radials = []  # Expected to be populated by subclasses


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        radii = np.linalg.norm(r, axis=1)

        # Evaluate each radial function for all radii
        # Each radial function returns shape (N,)
        # We want final shape (N, M)
        results = [func(radii) for func in self.radials]

        if not results:
             return np.zeros((r.shape[0], 0))

        return np.stack(results, axis=1)


    def __len__(self):
        return len(self._nl_pairs)

    def estimate_cutoff(self, threshold: float = 1e-6) -> float:
        raise NotImplementedError

    def lexographic_to_running_index(self, index: tuple) -> int:
        n, l = index
        if (n, l) not in self._nl_to_idx:
            raise ValueError(f"Invalid radial index (n={n}, l={l}) for n_max={self.n_max}.")
        return self._nl_to_idx[(n, l)]


    def running_to_lexographic_index(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self._nl_pairs):
            raise IndexError(f"Radial running index out of range: {idx}")
        return self._nl_pairs[idx]


class AngularFunctions(RIFunctions):
    """Base class for all angular functions

    AnbgularFunctions represents a set of angular functions, {Y_l^m}}, which are identified by their minor and magnetic quantum numbers, l and m.
    """

    def __init__(self, species: str, origin: tuple[float, float, float], l_max: int):
        super().__init__(species, origin)
        self.l_max = l_max

    def __call__(self, r: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    def __len__(self):
        return sum(2 * l + 1 for l in range(self.l_max + 1))


    def lexographic_to_running_index(self, index: tuple) -> int:
        l, m = index
        return sum(2 * l_ + 1 for l_ in range(l)) + (m + l)


    def running_to_lexographic_index(self, idx: int) -> tuple:
        l = 0
        count = 0
        while count + (2 * l + 1) <= idx:
            count += 2 * l + 1
            l += 1
        m = idx - count
        return l, m - l


class RIBasis(RIFunctions):
    """Class representing a basis for the resolution of the identity (RI).

    The class represents a set of functions X_{nl}^m, where each such function is a product of a radial component, R_{nl}, and
    a angular component, Y_l^m. This set is centered around an origin, R, and associated with a specific chemical element.

    When calling an RIBasis object with a set of N cartesian points {r}, it will return a (N, n_basis) numpy array, where n_basis is
    the number of basis functions in the RI basis. The returned array contains the numerical evaluations of X_{nl}^m(r) in lexographic
    order (n,l,m), where n runs slowest and m fastest. 

    Example:
    If n_max=2 and l_max=2, the RIBasis is a set of 18 X-functions. Calling ribasis([x,y,z]) will return a np.array containing the results
    of [[X_{10}^0, X_{11}^-1, X_{11}^0, X_{11}^1, X_{12}^-2, X_{12}^-1 ... X_{12}^2, X_{20}^0, X_{21}^-1, ... , X_{22}^2]]

    Parameters:
    -----------
        species (str): Chemical species associated with the basis
        origin (tuple[float, float, float]): Origin of all basis functions
        n_max (int | List[int]): Maximum major quantum number for basis functions
            As int, same number is used for all l
            As list, each l can be assigned its own n_max
        l_max (int): Maximum minor quantum number for basis functions

        radial_cls: Class of RadialFunctions, implementing the radial part of the basis
        angular_cls: Class of AngularFunctions, implementing the angular part of the basis
        radial_kwargs: Key-word arguments passed to RadialFunctions
        angular_kwards: Key-word arguments passed to AngularFunction
    """

    def __init__(self, species: str, origin: tuple[float, float, float], n_max: int | List[int], l_max: int,
                 radial_cls: type[RadialFunctions], angular_cls: type[AngularFunctions], cutoff: Cutoff,
                 radial_kwargs: dict = None, angular_kwargs: dict = None,
                 cell_vectors: np.ndarray = None):
        super().__init__(species, origin)

        self.n_max = normalize_n_max(n_max, l_max)
        self.l_max = l_max
        self._nl_pairs = enumerate_nl_pairs(self.n_max)
        self._nl_to_idx = {nl: idx for idx, nl in enumerate(self._nl_pairs)}
        self._angular_offsets = np.cumsum([0] + [2 * l + 1 for l in range(self.l_max + 1)])

        block_sizes = [2 * l + 1 for _, l in self._nl_pairs]
        self._basis_offsets = np.cumsum([0] + block_sizes)
        self._basis_size = int(self._basis_offsets[-1])

        self.radial_kwargs = radial_kwargs or {}
        self.angular_kwargs = angular_kwargs or {}

        self.cell_vectors = cell_vectors
        
        # Instantiate radial and angular components
        self.radial_funcs = radial_cls(species, origin, self.n_max, l_max, **self.radial_kwargs)
        self.angular_funcs = angular_cls(species, origin, l_max, **self.angular_kwargs)

        self.cutoff = self._resolve_cutoff(cutoff)
        self.lattice_vectors = self._get_lattice_vectors()


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        total_val = np.zeros((r.shape[0], self._basis_size))

        for G in self.lattice_vectors:
            # __call__ takes absolute coordinates; each periodic image is centered at origin + G.
            r_shifted = r - self.origin - G
            rad_vals = self.radial_funcs(r_shifted)  # Shape (N, n_rad_pairs)
            ang_vals = self.angular_funcs(r_shifted) # Shape (N, n_ang_funcs)

            for radial_idx, (_, l) in enumerate(self._nl_pairs):
                angular_start = self._angular_offsets[l]
                angular_end = self._angular_offsets[l + 1]
                basis_start = self._basis_offsets[radial_idx]
                basis_end = self._basis_offsets[radial_idx + 1]

                total_val[:, basis_start:basis_end] += (
                    rad_vals[:, radial_idx:radial_idx + 1] * ang_vals[:, angular_start:angular_end]
                )
        
        return total_val


    def __len__(self):
        return self._basis_size


    def lexographic_to_running_index(self, index: tuple) -> int:
        n, l, m = index
        if (n, l) not in self._nl_to_idx:
            raise ValueError(f"Invalid basis index (n={n}, l={l}) for n_max={self.n_max}.")
        if m < -l or m > l:
            raise ValueError(f"Invalid magnetic index m={m} for l={l}.")

        pair_idx = self._nl_to_idx[(n, l)]
        return int(self._basis_offsets[pair_idx] + (m + l))


    def running_to_lexographic_index(self, idx: int) -> tuple:
        if idx < 0 or idx >= self._basis_size:
            raise IndexError(f"Basis running index out of range: {idx}")

        pair_idx = int(np.searchsorted(self._basis_offsets, idx, side="right") - 1)
        n, l = self._nl_pairs[pair_idx]
        local_m = idx - int(self._basis_offsets[pair_idx])
        return n, l, local_m - l


    def _resolve_cutoff(self, cutoff: Cutoff) -> float | None:
        if cutoff == CutoffType.NON_PERIODIC:
            return None
        elif cutoff == CutoffType.ESTIMATE:
            return self.radial_funcs.estimate_cutoff()
        elif cutoff == CutoffType.FIRST_NEIGHBOURS:
            if self.cell_vectors is None or abs(np.linalg.det(self.cell_vectors)) < 1e-12:
                raise ValueError("Cell vectors required for FIRST_NEIGHBOURS cutoff.")

            shifts = np.array(
                [[i, j, k] for i in (-1, 0, 1)
                 for j in (-1, 0, 1)
                 for k in (-1, 0, 1)]
            )
            vectors = shifts @ self.cell_vectors
            return float(np.max(np.linalg.norm(vectors, axis=1))) + 1e-10
        elif isinstance(cutoff, (int, float)):
            return float(cutoff)
        raise TypeError(f"Invalid cutoff type: {type(cutoff)}")


    def _get_lattice_vectors(self) -> np.ndarray:
        if self.cutoff is None:
            return np.array([[0, 0, 0]])

        if self.cell_vectors is None or np.linalg.det(self.cell_vectors) == 0:
            raise ValueError("Cell vector is singular, even though cell is treated as periodic (cutofftype is not NON_PERIODIC)")

        inv_cell = np.linalg.inv(self.cell_vectors)
        max_indices = np.ceil(self.cutoff * np.linalg.norm(inv_cell, axis=0)).astype(int)
        
        n1, n2, n3 = np.mgrid[-max_indices[0]:max_indices[0]+1, -max_indices[1]:max_indices[1]+1, -max_indices[2]:max_indices[2]+1]
        lattice_points = np.vstack([n1.ravel(), n2.ravel(), n3.ravel()]).T
        
        vectors = np.dot(lattice_points, self.cell_vectors)
        
        return vectors[np.linalg.norm(vectors, axis=1) <= self.cutoff]


class RIBasisSet():
    """Class representing a complete RI basis for a structure, that is, a list of RIBasis objects for every atom

    Parameters
    ----------

    rho_origin_file: str
        Path to file from which electronic density is generated (e.g. .cube file)
    specifications: dict | str
        A dictionary mapping chemical species to their basis specifications. The basis specifications correspond
        directly to the parameters of RIBasis, except for species and origin which are determined from the .cube file.
        For example:

        {
            "O": {
                "n_max": 2,
                "l_max": 2,
                "radial_method": "gaussian",
                "angular_method": "real_spherical",
                "radial_params": {"alphas": [0.5, 1.0]},
            },
        }

    The dictionary can be provided as a path to as .json file as well.

    ribasis_loader: Callable
        A function that takes the parameters (species, origin, n_max, l_max, radial_method, angular_method, radial_kwargs,
        angular_kwargs) and returns an RIBasis object.

    cutoff: float | None
        Cutoff up to which periodicity of crystal is computed. When calling RIBasisSet, all basis functions within
        the cutoff contribute to the computed results.

        If set to None, the basis set will be treated as non-periodic.

    order_by_species: bool
        Whether to order the RIBasis objects in the final list by species. If False, the ordering will be the same as in
        the rho origin file.

    Methods:
    --------

    __call__(r: np.ndarray) -> np.ndarray
        Calling the RIBasisSet with a set of cartesian points will return a block diagonal array containing the
        individual evaluations of the RIBasis functions for each atom.

    """

    def __init__(self, rho_origin_file: str, specifications: dict | str,
                 ribasis_loader: Callable, cutoff: Cutoff, order_by_species: bool = False):
        
        if isinstance(specifications, str):
            with open(specifications, 'r') as f:
                specifications = json.load(f)
                
        self.specifications = specifications
        self.cell_vectors, self.species_and_positions = self._load_structure(rho_origin_file, return_ordered=order_by_species)
        self.ribases = []
        self.cutoff = cutoff
        self.loader_func = ribasis_loader

        for species, position in self.species_and_positions:
            if species not in specifications:
                raise ValueError(f"Species '{species}' found in structure file but not in specifications.")
            ribasis = self._load_ribasis(species, position)
            self.ribases.append(ribasis)


    def __call__(self, r: np.ndarray) -> np.ndarray:
        if r.ndim != 2 or r.shape[1] != 3:
            raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

        # Pass absolute coordinates to each basis; RIBasis handles origin internally.
        results = [ribasis(r) for ribasis in self.ribases]
        return np.hstack(results)


    def __len__(self) -> int:
        return len(self.ribases)


    def __iter__(self) -> Iterator[RIBasis]:
        return iter(self.ribases)


    def __getitem__(self, item):
        return self.ribases[item]


    def span(self, coefficients: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Returns a function that evaluates the linear combination of RIBasis functions defined by the input coefficients.

        Parameters:
        -----------
        coefficients: array-like
            A 1D array of coefficients, with length equal to the total number of RIBasis functions in the set.

        Returns:
        --------
        A callable function that takes a (N, 3) array of cartesian points and returns a (N,) array of evaluations of the linear combination of RIBasis functions at those points.
        """
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != 1 or coefficients.shape[0] != sum(len(ribasis) for ribasis in self.ribases):
            raise ValueError(f"Coefficients must be a 1D array with length equal to the total number of RIBasis functions ({sum(len(ribasis) for ribasis in self.ribases)}), got shape {coefficients.shape}")

        def linear_combination(r: np.ndarray) -> np.ndarray:
            r = np.ascontiguousarray(r)
            if r.ndim != 2 or r.shape[1] != 3:
                raise ValueError(f"Input must be an array of shape (N, 3), got {r.shape}")

            # Keep the same absolute-coordinate convention as RIBasisSet.__call__.
            results = [ribasis(r) for ribasis in self.ribases]
            combined = np.hstack(results)
            return combined @ coefficients

        return linear_combination


    @staticmethod
    def _load_structure(rho_origin_file: str, return_ordered: bool) \
            -> Tuple[np.ndarray, List[Tuple[str, Tuple[float, float, float]]]]:
        """Loads the structure file using ASE, returns the untit cell vectors and list of atom species with position"""

        atoms = read(str(rho_origin_file))
        cell_vectors = atoms.cell
        species_list: List[Tuple[str, Tuple[float, float, float]]] = [(str(atom.symbol), tuple(atom.position)) for atom in atoms]

        if return_ordered:
            species_list.sort(key=lambda x: (x[0], x[1][2], x[1][0], x[1][1]))

        return cell_vectors, species_list


    def _load_ribasis(self, species: str, origin: tuple[float, float, float]) -> RIBasis:
        specs = self.specifications[species]
        return self.loader_func(
            species=species,
            origin=origin,
            cell_vectors=self.cell_vectors,
            cutoff=self.cutoff,
            **specs
        )
