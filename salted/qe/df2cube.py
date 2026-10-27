"""Reconstruct the real-space electron density from fitted RI-basis coefficients, and compare it
against the reference QE cube -- the QE analogue of `salted.cp2k.df2cube`'s "e_density" output.

This is the actual quality check for a density fit: the diagnostics in `salted.qe.density_fitting`
(charge conservation, overlap conditioning, coefficient magnitude) are all self-consistency checks
on the linear algebra, and would pass even for a basis that doesn't represent the true density well.
This module evaluates sum_i c_i * chi_i(r) on the same grid as `conf_N/rho.cube` and reports the
density reconstruction error as a percentage of the total charge, exactly as CP2K's own pipeline does.

Reuses the same radial-function normalization and real-space evaluation CP2K's `df2cube.py` uses --
pure math on (lmax, nmax, alphas, contranorm) plus a cube grid, nothing CP2K-specific -- just adapted
to read `conf_N/rho.cube` (`salted.qe.qe2cube`'s naming) and `coefficients_conf{N}.npy`
(`salted.qe.density_fitting`'s output) instead of being driven by an external caller.
"""

import argparse
import math
import os
import os.path as osp

import numpy as np
from scipy import special
from scipy.interpolate import interp1d

from salted import basis
from salted.constants import bohr2angs
from salted.qe.qe2cube import CUBE_FNAME
from salted.qe.xyz2sys import STRUCTURE_FNAME
from salted.sys_utils import ARGHELP_INDEX_STR, ParseConfig, parse_index_str

from ase.io import read as ase_read

RECONSTRUCTED_CUBE_SUFFIX = "_reconstructed.cube"


def _read_cube(cube_path: str, natoms: int):
    with open(cube_path) as cubefile:
        lines = cubefile.readlines()
    nside = [int(lines[3].split()[0]), int(lines[4].split()[0]), int(lines[5].split()[0])]
    npoints = nside[0] * nside[1] * nside[2]
    dx, dy, dz = float(lines[3].split()[1]), float(lines[4].split()[2]), float(lines[5].split()[3])
    origin = np.asarray(lines[2].split(), dtype=float)[1:4]
    rho = np.fromstring("".join(lines[6 + natoms:]), dtype=np.float64, sep=" ")
    if rho.size != npoints:
        raise ValueError(f"{cube_path}: inconsistent number of grid points ({rho.size} != {npoints})")
    return rho, nside, (dx, dy, dz), origin


def _periodic_offsets(periodicity: str, repmax: np.ndarray):
    """Lattice-image offsets (in units of lattice vectors) to sum over for a given periodicity."""
    if periodicity == "3D":
        return [
            (ix, iy, iz)
            for ix in range(-repmax[0], repmax[0] + 1)
            for iy in range(-repmax[1], repmax[1] + 1)
            for iz in range(-repmax[2], repmax[2] + 1)
        ]
    elif periodicity == "2D":
        return [(ix, iy, 0) for ix in range(-repmax[0], repmax[0] + 1) for iy in range(-repmax[1], repmax[1] + 1)]
    elif periodicity == "0D":
        return [(0, 0, 0)]
    else:
        raise ValueError(f"Unsupported periodicity: {periodicity}")


def _real_spherical_harmonics(l: int, lth: np.ndarray, lph: np.ndarray) -> np.ndarray:
    """Real spherical harmonics Y_l^m for m=-l..l, same convention as salted.cp2k.df2cube."""
    ylm_real = np.zeros((2 * l + 1, len(lth)))
    lm = 0
    for m in range(-l, 1):
        ylm = special.sph_harm_y(l, m, lth, lph)
        if m == 0:
            ylm_real[lm, :] = np.real(ylm) / np.sqrt(2.0)
            lm += l + 1
        else:
            ylm_real[lm, :] = -np.imag(ylm)
            ylm_real[lm - 2 * m, :] = np.real(ylm)
            lm += 1
    ylm_real *= np.sqrt(2.0)
    return ylm_real


def reconstruct_one(inp, geom_idx: int) -> tuple[float, float]:
    """Reconstruct conf `geom_idx`'s density from its fitted coefficients, compare to its reference
    cube, write the reconstructed density to `saltedpath/cubes/`, and return (n_electrons, %MAE).
    """
    dfbasis = inp.qm.dfbasis
    species = inp.system.species
    lmax, nmax = basis.basiset(dfbasis)
    lmax_max = max(lmax[spe] for spe in species)

    conf_dir = osp.join(inp.qm.path2qm, f"conf_{geom_idx + 1}")
    atoms = ase_read(osp.join(conf_dir, STRUCTURE_FNAME))
    atomic_symbols = atoms.get_chemical_symbols()
    valences = atoms.get_atomic_numbers()
    natoms = len(atomic_symbols)
    coords = atoms.get_positions() / bohr2angs
    cell = atoms.get_cell() / bohr2angs

    coefs = np.load(osp.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{geom_idx}.npy"))

    naux = sum(nmax[(atomic_symbols[iat], l)] * (2 * l + 1) for iat in range(natoms) for l in range(lmax[atomic_symbols[iat]] + 1))
    if naux != len(coefs):
        raise ValueError(f"conf {geom_idx + 1}: inconsistent number of coefficients and naux: ncoefs={len(coefs)}, naux={naux}")

    # Read auxiliary basis info and build normalized radial interpolants, same formula as
    # salted.cp2k.df2cube / salted.qe.get_basis_info's own round-trip (inner = self-overlap of the
    # contracted primitive Gaussians).
    bdir = osp.join(inp.salted.saltedpath, "basis")
    alphas, contra, rcuts = {}, {}, {}
    for spe in species:
        for l in range(lmax[spe] + 1):
            alphas[(spe, l)] = np.atleast_1d(np.loadtxt(osp.join(bdir, f"{spe}-{dfbasis}-alphas-L{l}.dat")))
            contra[(spe, l)] = np.atleast_2d(np.loadtxt(osp.join(bdir, f"{spe}-{dfbasis}-contra-L{l}.dat")))
            npgf = alphas[(spe, l)].shape[0]
            rcuts_primitive = np.sqrt((2.0 + l) / (2.0 * alphas[(spe, l)])) * 4.0
            rcuts[(spe, l)] = float(np.max(rcuts_primitive))

    ngrid = 20000
    interp_radial = {}
    for spe in species:
        for l in range(lmax[spe] + 1):
            dxx = rcuts[(spe, l)] / float(ngrid - 1)
            npgf = len(alphas[(spe, l)])
            for n in range(nmax[(spe, l)]):
                inner = 0.0
                for ipgf1 in range(npgf):
                    for ipgf2 in range(npgf):
                        inner += (
                            contra[(spe, l)][n, ipgf1] * contra[(spe, l)][n, ipgf2] * 0.5
                            * special.gamma(l + 1.5) / ((alphas[(spe, l)][ipgf1] + alphas[(spe, l)][ipgf2]) ** (l + 1.5))
                        )
                rvec = np.arange(ngrid) * dxx
                radial = np.zeros(ngrid)
                for ipgf in range(npgf):
                    radial += contra[(spe, l)][n, ipgf] * rvec ** l * np.exp(-alphas[(spe, l)][ipgf] * rvec ** 2)
                radial /= np.sqrt(inner)
                interp_radial[(spe, l, n)] = interp1d(rvec, radial, kind="quadratic")

    rho_qm, nside, (dx, dy, dz), origin = _read_cube(osp.join(conf_dir, CUBE_FNAME), natoms)
    nx, ny, nz = nside
    npoints = nx * ny * nz

    grid_regular = np.transpose(
        np.asarray(np.meshgrid(dx * np.arange(nx), dy * np.arange(ny), dz * np.arange(nz))), (2, 1, 3, 0)
    ).reshape((npoints, 3))

    rhor = np.zeros(npoints)
    iaux = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(lmax[spe] + 1):
            repmax = np.zeros(3, int)
            if inp.qm.periodic != "0D":
                for ix in range(3):
                    nreps = math.ceil(rcuts[(spe, l)] / cell[ix, ix]) if cell[ix, ix] > 0 else 0
                    repmax[ix] = max(nreps, 1)
            offsets = _periodic_offsets(inp.qm.periodic, repmax)

            for n in range(nmax[(spe, l)]):
                coef = coefs[iaux:iaux + 2 * l + 1]
                for ix, iy, iz in offsets:
                    coord = coords[iat] - origin
                    coord[0] += ix * cell[0, 0]
                    coord[1] += iy * cell[1, 1]
                    coord[2] += iz * cell[2, 2]
                    dvec = grid_regular - coord
                    d2list = np.sum(dvec ** 2, axis=1)
                    indx = np.where(d2list <= rcuts[(spe, l)] ** 2)[0]
                    rr = dvec[indx]
                    lr = np.sqrt(np.sum(rr ** 2, axis=1)) + 1e-12
                    lth = np.arccos(rr[:, 2] / lr)
                    lph = np.arctan2(rr[:, 1], rr[:, 0])
                    ylm_real = _real_spherical_harmonics(l, lth, lph)
                    radial_gto = interp_radial[(spe, l, n)](lr)
                    gto = np.einsum("ab,b->ab", ylm_real, radial_gto)
                    rhor[indx] += np.dot(coef, gto)
                iaux += 2 * l + 1

    nele = float(np.sum(rhor) * dx * dy * dz)
    error_pct = float(np.sum(np.abs(rhor - rho_qm)) * dx * dy * dz / nele * 100.0) if nele != 0 else float("nan")

    dirpath = osp.join(inp.salted.saltedpath, "cubes")
    os.makedirs(dirpath, exist_ok=True)
    out_path = osp.join(dirpath, f"conf_{geom_idx + 1}{RECONSTRUCTED_CUBE_SUFFIX}")
    with open(out_path, "w") as cubef:
        print("Reconstructed electron density", file=cubef)
        print("CUBE FORMAT", file=cubef)
        print(natoms, origin[0], origin[1], origin[2], file=cubef)
        metric = np.diag([dx, dy, dz])
        for ix in range(3):
            print(nside[ix], metric[ix, 0], metric[ix, 1], metric[ix, 2], file=cubef)
        for iat in range(natoms):
            print(valences[iat], float(valences[iat]), coords[iat][0], coords[iat][1], coords[iat][2], file=cubef)
        for v in rhor:
            print(v, file=cubef)

    return nele, error_pct


def main(geom_indexes: list[int] | None):
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    ndata = len(ase_read(inp.system.filename, ":"))
    if geom_indexes is None:
        geom_indexes = list(range(ndata))
    else:
        geom_indexes = [i for i in geom_indexes if i < ndata]

    for geom_idx in geom_indexes:
        try:
            nele, error_pct = reconstruct_one(inp, geom_idx)
            print(f"conf {geom_idx + 1}: integrated charge = {nele:.6f}, density reconstruction error = {error_pct:.4f}% of total charge")
        except Exception as exc:
            print(f"conf {geom_idx + 1}: FAILED: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=str, default="all", help=ARGHELP_INDEX_STR)
    args = parser.parse_args()

    main(parse_index_str(args.idx))
