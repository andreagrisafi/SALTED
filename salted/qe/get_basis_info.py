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
      "n_max": 3,
      "l_max": 2,
      "radial_method": "atomic",
      "angular_method": "real_spherical",
      "radial_params": {"filename": "./atomic_densities/O_TZP_rc10.0_qe.dat"},
      "n_gauss": 6
    }

`n_gauss` (optional, default `DEFAULT_N_GAUSS`) sets how many primitive Gaussians are fitted per
(n, l) radial channel.

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

DEFAULT_N_GAUSS = 6


def build(dryrun: bool = False, force_overwrite: bool = False):
    """Scheme: for each species, read its `{species}-{dfbasis}.json` RI-basis specification,
    fit it to Gaussians, and write the result to the basis database and basis .dat files.
    """
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    lmax, nmax, alphas, contra = fit_basis_info(inp.system.species, inp.qm.dfbasis)

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


def fit_basis_info(species: list[str], dfbasis: str) -> tuple[
    dict[str, int], dict[tuple[str, int], int], dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]
]:
    """Build and Gaussian-fit the atomic RI basis for each species.

    File naming convention: `{species}-{dfbasis}.json`, read from the current working directory.
    No file IO besides reading those specification files; see module docstring for format and
    for the block-diagonal convention used to reconcile independently-fitted (n, l) channels with
    the shared alphas/contra file format.
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
        fitted = interpolate_atomic_basis(ribasis, species=spe, position=(0.0, 0.0, 0.0), num_max=n_gauss)

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
