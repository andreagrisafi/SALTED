"""Translate a Gaussian-fitted atomic RI basis into SALTED basis info, for use with Quantum ESPRESSO.

Unlike CP2K, Quantum ESPRESSO produces real-space densities (.cube files) but no RI basis guess.
The `salted.qe` subpackage builds one instead: it reads atomic radial wavefunctions, forms a product
basis, compresses it by diagonalizing the overlap matrix (`salted.qe.atomic.Atomic_Rad`), and fits
each resulting radial function to a sum of primitive Gaussians
(`salted.qe.interpolate_gaussian.interpolate_atomic_basis`). This module writes that fit to disk in
the same format SALTED already uses for CP2K/AIMS/PySCF basis sets (alphas/contraction .dat files
plus the `basis_data.yaml` entry via `BasisClient`), so the rest of the pipeline can consume it
unchanged.

File naming convention (mirrors CP2K, see `salted.cp2k.get_basis_info`): for each species, a basis
specification file `{species}-{dfbasis}.json` must exist in the current working directory. It is the
basis specification consumed by `salted.qe.load_basis`, e.g.:

    {
      "l_max": 2,
      "radial_method": "atomic",
      "angular_method": "real_spherical",
      "radial_params": {
        "filename": "./atomic_densities/O_TZP_rc10.0_qe.dat",
        "n_max": [4, 4, 3],
        "energy_file": "./atomic_densities/O_TZP_rc10.0_eigenvalues.dat",
        "energy_scale": 1.0
      },
      "n_gauss": 6
    }

`n_gauss` (optional, default `DEFAULT_N_GAUSS`) sets how many primitive Gaussians are fitted per
(n, l) radial channel.

For the `"atomic"` radial method, `n_max` (required; `radial_params`, int or one value per l) sets
exactly how many compressed radial functions to keep per l: the n_max[l] eigenvectors of that l's
(generalized, when energy_file is given) overlap-matrix eigenproblem with the largest eigenvalue.
There is no automatic eigenvalue cutoff: the atomic self-multiplication overlap matrix at a given l
only has as many independent eigenvectors as the data supports, and asking for more than that
returns near-zero-eigenvalue noise, which can make the downstream (density-fitting) overlap matrix
ill-conditioned/singular or even produce NaN coefficients from Gaussian-fitting a near-zero-norm
function -- pick n_max conservatively per l (e.g. guided by
`salted.qe.density_fitting.diagnose_overlap_conditioning` on a prior run) rather than relying on an
automatic cutoff. See `salted.qe.atomic.Atomic_Rad._compress_basis`.

`energy_file`/`energy_scale` (both optional; `radial_params`, "atomic" method only) bias compression
toward low-energy input orbitals rather than pure overlap/completeness -- see
`salted.qe.atomic.Atomic_Rad._compress_basis`. Omit them to keep the original, purely
overlap-based compression.

Each (n, l) radial channel is fitted independently, with its own Gaussian exponents, so exponents
are not shared across n the way a single CP2K basis "set" is. To stay compatible with the shared
`{species}-{dfbasis}-alphas-L{l}.dat` / `-contra-L{l}.dat` format, the per-n exponents for a given l
are concatenated into one primitive list and the contraction matrix is block-diagonal: row n is zero
outside the columns belonging to its own fit. This is the same block-diagonal structure CP2K itself
uses when a basis has multiple exponent "sets" for the same l.
"""

import json
import os
import os.path as osp

import numpy as np

from salted.basis_client import BasisClient, SpeciesBasisData
from salted.get_basis_info import get_parser
from salted.sys_utils import ParseConfig

from salted.qe.loader import load_basis
from salted.qe.interpolate_gaussian import interpolate_atomic_basis
from salted.qe.qe_types import CutoffType
from salted.sys_utils import PLACEHOLDER

DEFAULT_N_GAUSS = 6
DEFAULT_ALPHA_THRESHOLD = 1e-3


def max_safe_alpha(inp, threshold: float = DEFAULT_ALPHA_THRESHOLD) -> float:
    """Largest Gaussian exponent whose reciprocal-space weight has decayed below `threshold` (of
    its on-axis peak) by the density grid's own Nyquist wavevector -- i.e. the largest exponent
    density_fitting.py's reciprocal-space fit can actually resolve on that grid, derived from
    `ecutwfc`/`ecutrho` (Ry) alone, since no cube file exists yet at basis-generation time.

    G_max (bohr^-1) = sqrt(ecutrho), QE's standard Ry-to-bohr^-1 plane-wave relation; ecutrho
    defaults to QE's own norm-conserving-pseudopotential convention (4*ecutwfc) when not set.
    Solving exp(-G_max^2/(4*alpha)) = threshold for alpha gives the bound below.

    Exponents above this bound are only band-limited (not rejected) by density_fitting.py's own
    machinery when actually used for fitting, so a function fit with such an exponent is
    self-consistent within that reciprocal-space fit but mismatches an exact real-space
    reconstruction of the same (alpha, contra) (e.g. salted.qe.df2cube) -- see fit_gaussians'
    alpha_max docstring (salted.qe.interpolate_gaussian).
    """
    ecutrho = inp.qm.ecutrho
    if ecutrho == PLACEHOLDER:
        ecutrho = 4.0 * inp.qm.ecutwfc
    g_max = np.sqrt(ecutrho)
    return g_max**2 / (4.0 * np.log(1.0 / threshold))


def build(dryrun: bool = False, force_overwrite: bool = False):
    """Scheme: for each species, read its `{species}-{dfbasis}.json` RI-basis specification,
    fit it to Gaussians, and write the result to the basis database and basis .dat files.
    """
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    alpha_max = max_safe_alpha(inp)
    print(f"Capping fitted Gaussian exponents at alpha_max={alpha_max:.3g} bohr^-2 "
          f"(derived from ecutwfc/ecutrho, so fits stay resolvable on the density-fitting grid)")
    lmax, nmax, alphas, contra = fit_basis_info(inp.system.species, inp.qm.dfbasis, alpha_max)

    """Convert to basis_client format"""
    basis_data: dict[str, SpeciesBasisData] = {}
    for spe in lmax.keys():
        basis_data[spe] = {
            "lmax": lmax[spe],
            "nmax": [nmax[(spe, l)] for l in range(lmax[spe] + 1)],
        }

    # Generate directory for saving basis set info
    bdir = osp.join(inp.salted.saltedpath, "basis")
    if not osp.exists(bdir):
        os.mkdir(bdir)

    """write to the database"""
    if dryrun:
        print("Dryrun mode, not writing to the database")
        print(f"{inp.system.species=}")
        print(f"{inp.qm.dfbasis=}")
        print(f"{lmax=}, {nmax=}")
        print(f"{basis_data=}")
        print(f"{alphas=}")
    else:
        for spe in inp.system.species:
            for l in range(lmax[spe] + 1):
                np.savetxt(osp.join(bdir, f"{spe}-{inp.qm.dfbasis}-alphas-L{l}.dat"), alphas[(spe, l)])
                np.savetxt(osp.join(bdir, f"{spe}-{inp.qm.dfbasis}-contra-L{l}.dat"), contra[(spe, l)])
        BasisClient().write(inp.qm.dfbasis, basis_data, force_overwrite)


def fit_basis_info(species: list[str], dfbasis: str, alpha_max: float = 1e6) -> tuple[
    dict[str, int], dict[tuple[str, int], int], dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]
]:
    """Build and Gaussian-fit the atomic RI basis for each species.

    File naming convention: `{species}-{dfbasis}.json`, read from the current working directory.
    No file IO besides reading those specification files; see module docstring for format and
    for the block-diagonal convention used to reconcile independently-fitted (n, l) channels with
    the shared alphas/contra file format.

    alpha_max (see `max_safe_alpha`) bounds how sharp a fitted Gaussian primitive is allowed to be,
    so the fit never produces exponents the actual density-fitting grid can't resolve.
    """
    lmax: dict[str, int] = {}
    nmax: dict[tuple[str, int], int] = {}
    alphas: dict[tuple[str, int], np.ndarray] = {}
    contra: dict[tuple[str, int], np.ndarray] = {}

    for spe in species:
        spec_file = f"{spe}-{dfbasis}.json"
        with open(spec_file) as f:
            spec = json.load(f)
        n_gauss = spec.pop("n_gauss", DEFAULT_N_GAUSS)

        ribasis = load_basis(species=spe, origin=(0.0, 0.0, 0.0), cutoff=CutoffType.NON_PERIODIC, **spec)
        fitted = interpolate_atomic_basis(ribasis, species=spe, position=(0.0, 0.0, 0.0), num_max=n_gauss, alpha_max=alpha_max)

        lmax[spe] = fitted.l_max
        for l, n_l in enumerate(fitted.n_max):
            nmax[(spe, l)] = n_l
            if n_l == 0:
                alphas[(spe, l)] = np.zeros(0)
                contra[(spe, l)] = np.zeros((0, 0))
                continue

            # n in fitted.alphas/.coeffs is 1-indexed (see core.enumerate_nl_pairs); rows of the
            # contra matrix are 0-indexed, matching how nmax is used everywhere else in SALTED.
            per_n_alphas = [fitted.alphas[(n, l)] for n in range(1, n_l + 1)]
            per_n_coeffs = [fitted.coeffs[(n, l)] for n in range(1, n_l + 1)]
            npgf_total = sum(len(a) for a in per_n_alphas)

            l_contra = np.zeros((n_l, npgf_total))
            col = 0
            for row, (a_n, c_n) in enumerate(zip(per_n_alphas, per_n_coeffs)):
                l_contra[row, col:col + len(a_n)] = c_n
                col += len(a_n)

            alphas[(spe, l)] = np.concatenate(per_n_alphas)
            contra[(spe, l)] = l_contra

    return lmax, nmax, alphas, contra


if __name__ == "__main__":
    print("Please call `python -m salted.get_basis_info` instead of this file")

    parser = get_parser()
    args = parser.parse_args()

    build(dryrun=args.dryrun, force_overwrite=args.force_overwrite)
