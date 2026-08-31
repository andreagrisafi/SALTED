from typing import Type, Any, Dict, Optional
from .core import RIBasisSet, RIBasis, RadialFunctions, AngularFunctions
from .gaussian import PrimitiveGaussianRadials
from .real_spher_harmonic import RealSphericalHarmonics
from .qe_types import Cutoff, CutoffType

RADIAL_REGISTRY: Dict[str, Type[RadialFunctions]] = {}

ANGULAR_REGISTRY: Dict[str, Type[AngularFunctions]] = {}

def register_radial(name: str, cls: Type[RadialFunctions]):
    """Register a new radial function implementation."""
    RADIAL_REGISTRY[name] = cls


def register_angular(name: str, cls: Type[AngularFunctions]):
    """Register a new angular function implementation."""
    ANGULAR_REGISTRY[name] = cls


def load_basis(species: str,
               origin: tuple[float, float, float],
               n_max: int | list[int],
               l_max: int,
               radial_method: str,
               angular_method: str,
               cutoff: Cutoff,
               radial_params: Optional[Dict[str, Any]] = None,
               angular_params: Optional[Dict[str, Any]] = None,
               cell_vectors = None,
               ) -> RIBasis:
    """
    Load and initialize an RI Basis set based on the specified methods and parameters.

    Parameters
    ----------
    species : str
        Chemical species label (e.g. 'O', 'H').
    origin : tuple[float, float, float]
        Center of the basis functions.
    n_max : int | list[int]
        Maximum radial quantum number(s), interpreted as counts per angular momentum channel.
        If int, the same count is used for all l in [0, l_max].
        If list, ``n_max[l]`` is used for each l and the list must have length ``l_max + 1``.
    l_max : int
        Maximum angular momentum quantum number (inclusive).
    radial_method : str, optional
        Name of the radial function method to use (default: 'gaussian').
    angular_method : str, optional
        Name of the angular function method to use (default: 'spherical_harmonics').
    radial_params : dict, optional
        Additional parameters to pass to the radial function constructor (e.g. {'alphas': [...]}).
    angular_params : dict, optional
        Additional parameters to pass to the angular function constructor.

    Returns
    -------
    RIBasis
        The constructed RI Basis object.

    Raises
    ------
    ValueError
        If the specified radial or angular method is not registered.
    """

    radial_cls = RADIAL_REGISTRY.get(radial_method)
    if not radial_cls:
        raise ValueError(f"Unknown radial method: '{radial_method}'. Available: {list(RADIAL_REGISTRY.keys())}")

    angular_cls = ANGULAR_REGISTRY.get(angular_method)
    if not angular_cls:
        raise ValueError(f"Unknown angular method: '{angular_method}'. Available: {list(ANGULAR_REGISTRY.keys())}")

    return RIBasis(species, origin, n_max, l_max,
                   radial_cls, angular_cls,
                   radial_kwargs=radial_params,
                   angular_kwargs=angular_params,
                   cell_vectors=cell_vectors,
                   cutoff=cutoff)


def load_basis_set(rho_origin_file: str,
                   specifications: dict | str,
                   cutoff: Cutoff,
                   order_by_species: bool = False) -> RIBasisSet:
    """
    Load a complete RI basis set for a given structure.

    This function acts as a factory for creating an RIBasisSet object, injecting the
    necessary loader function to construct individual RIBasis instances.

    Parameters
    ----------
    rho_origin_file: str
        Path to file from which electronic density was created (e.g. .cube file)
    specifications : dict or str
        A dictionary or path to a JSON file with basis specifications for each species.
        Example for H2O (do not copy, as the alpha parameters are just arbitrary):
        {
            "H": {
                "n_max": 2,
                "l_max": 1,
                "radial_method": "gaussian",
                "angular_method": "real_spherical",
                "radial_params": {"alphas": [0.5, 1.0, 1.5, 2.0]},
            },
            "O": {
                "n_max": 2,
                "l_max": 2,
                "radial_method": "gaussian",
                "angular_method": "real_spherical",
                "radial_params": {"alphas": [0.5, 1.0, 1.5, 2.0]},
            },
        }
    cutoff : Cutoff
        The cutoff strategy for periodic images. Can be a float, "estimate", or "non-periodic".
    order_by_species : bool, optional
        If True, order the basis functions by species (default is False).

    Returns
    -------
    RIBasisSet
        The constructed RI basis set.
    """
    from .core import RIBasisSet  # Local import to avoid circular dependency
    return RIBasisSet(rho_origin_file, specifications, load_basis, cutoff=cutoff, order_by_species=order_by_species)
