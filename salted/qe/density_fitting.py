"""Project Quantum ESPRESSO electron densities onto the RI basis, producing SALTED training data.

Reuses `salted.cp2k.utils`'s Fourier-space density-fitting machinery as-is: it only depends on the
auxiliary (lmax, nmax, alphas, contranorm) RI basis info -- already written by
`salted.qe.get_basis_info` in the same `{species}-{dfbasis}-alphas-L{l}.dat` / `-contra-L{l}.dat`
format CP2K uses -- and on a real-space density grid (the cube file). Neither is CP2K-specific, so
none of that module needed changing; only two things did:

    - the cube file: `salted.qe.qe2cube` writes `conf_N/rho.cube`, not CP2K's
      `*ELECTRON_DENSITY-1_0.cube` naming. `read_cube` below mirrors CP2K's own cube parser
      byte-for-byte (the cube format itself is standard, shared between the two codes) against that
      filename.
    - per-species valence charge (`salted.cp2k.utils`'s "pseudocharge"), used only for the
      Lagrange-multiplier total-charge constraint. Read here from each UPF's `Z_valence` header
      field (`read_pseudocharge`) instead of CP2K's `{spe}-local_pseudo.dat` (a GTH-specific format).

`rloc` and the GTH-specific Hartree-energy diagnostic (`salted.cp2k.utils.compute_hartree_energy`)
are intentionally not ported: `rloc` is never used anywhere in the fitting computation itself (only
in that diagnostic), so there is nothing QE-equivalent to supply yet.

Requires `inp.qm.dfmetric` ("identity" or "coulomb", same meaning as for CP2K) and the
`salted.qe.run_qe` / `salted.qe.qe2cube` / `salted.qe.get_basis_info` outputs to already exist.
"""

import argparse
import os
import os.path as osp
import re
import time
from dataclasses import dataclass

import numpy as np

from salted.constants import bohr2angs
from salted.cp2k.utils import (
    build_contraction_matrix,
    build_gcutoff,
    get_basis_set_info_numba,
    get_reciprocal_grid,
    get_w_prim,
    gto_rec,
    gto_rec_ewald,
    gto_rec_g0,
    gto_rec_prim,
    overlap_coulomb_real,
    overlap_coulomb_rec,
    overlap_identity,
    pair_cutoffs,
    setup_pyscf_ewald,
    setup_pyscf_species,
)
from salted.qe.qe2cube import CUBE_FNAME
from salted.sys_utils import (
    ARGHELP_INDEX_STR,
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    parse_index_str,
    read_system,
)

UPF_Z_VALENCE_RE = re.compile(r'z_valence\s*=\s*"\s*([\d.eE+-]+)\s*"', re.IGNORECASE)


def read_pseudocharge(species: list[str], inp) -> dict[str, float]:
    """Read each species' valence charge from its UPF pseudopotential header (Z_valence)."""
    pseudo_map = inp.qm.pseudopotentials.to_dict()
    pseudocharge = {}
    for spe in species:
        upf_path = osp.join(inp.qm.pseudo_dir, pseudo_map[spe])
        with open(upf_path) as f:
            text = f.read()
        match = UPF_Z_VALENCE_RE.search(text)
        if match is None:
            raise ValueError(f"Could not find z_valence in {upf_path}")
        pseudocharge[spe] = float(match.group(1))
    return pseudocharge


def read_cube(conf_dir: str, natoms: int):
    """Read a Gaussian cube file's grid and density.

    Same raw parsing CP2K's own density_fitting.py does (grid spacing/origin in bohr, per the cube
    format convention) -- the cube format itself is standard and shared between CP2K and QE's pp.x.
    """
    cube_path = osp.join(conf_dir, CUBE_FNAME)
    with open(cube_path) as cubefile:
        header = [cubefile.readline() for _ in range(6 + natoms)]
        line_ori, line_x, line_y, line_z = (header[i].split() for i in range(2, 6))
        nx, dx = int(line_x[0]), float(line_x[1])
        ny, dy = int(line_y[0]), float(line_y[2])
        nz, dz = int(line_z[0]), float(line_z[3])
        origin = np.asarray(line_ori[1:4], dtype=float)
        npoints = nx * ny * nz
        rho = np.fromstring(cubefile.read(), dtype=np.float64, sep=" ")

    if rho.size != npoints:
        raise ValueError(f"{cube_path}: inconsistent number of grid points ({rho.size} != {npoints})")

    return rho.reshape((nx, ny, nz)), (nx, ny, nz), (dx, dy, dz), origin


@dataclass
class FitContext:
    """Everything that is the same across all configurations, computed once in `main`."""

    species: list[str]
    lmax: dict
    nmax: dict
    lmax_max: int
    atomic_symbols: list
    atomic_coords: list
    natoms: np.ndarray
    df_metric: str
    path2qm: str
    alphas: dict
    contranorm: dict
    npgf: dict
    lmax_numba: dict
    nmax_numba: dict
    nbasis: dict
    pyscf_data: object
    pyscf_ewald: object
    rcut_pairs: dict
    sigma_ewald: float | None
    pseudocharge: dict
    verbose: bool


def fit_one(ctx: FitContext, geom_idx: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit a single configuration's density onto the RI basis. Returns (coefficients, overlap)."""
    conf_dir = osp.join(ctx.path2qm, f"conf_{geom_idx + 1}")
    atomic_symbols = ctx.atomic_symbols[geom_idx]
    atomic_coords = ctx.atomic_coords[geom_idx]
    natoms = ctx.natoms[geom_idx]

    ntype = {spe: 0 for spe in ctx.species}
    n_elec = 0.0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        ntype[spe] += 1
        n_elec += ctx.pseudocharge[spe]
    ncoefs = sum(ctx.nbasis[spe] * ntype[spe] for spe in ctx.species)

    rho_KS, (nx, ny, nz), (dx, dy, dz), origin = read_cube(conf_dir, natoms)
    volume = (nx * dx) * (ny * dy) * (nz * dz)
    cell = np.diag([nx * dx, ny * dy, nz * dz])

    Gvec = get_reciprocal_grid(nx, ny, nz, dx, dy, dz)
    mask = (
        (Gvec[:, 2] > 0)
        | ((Gvec[:, 2] == 0) & (Gvec[:, 1] > 0))
        | ((Gvec[:, 2] == 0) & (Gvec[:, 1] == 0) & (Gvec[:, 0] >= 0))
    )
    Gvec_half = Gvec[mask]
    if ctx.df_metric == "coulomb":
        Gvec_half = Gvec_half[1:]  # exclude G=0

    knorm_vec = np.linalg.norm(Gvec_half, axis=1)
    sort_idx = np.argsort(knorm_vec)
    Gvec_half = Gvec_half[sort_idx]
    knorm_vec = knorm_vec[sort_idx]

    gcuts = build_gcutoff(ctx.alphas, ctx.species, ctx.lmax, knorm_vec)

    if ctx.df_metric == "coulomb":
        gmax_ewald = 2.0 * np.pi / ctx.sigma_ewald
        gmax_ewald_idx = np.searchsorted(knorm_vec, gmax_ewald).astype(np.int64)

    rho_KS_rec = np.fft.fftn(rho_KS).ravel() * dx * dy * dz
    rho_KS_rec = rho_KS_rec[mask]
    if ctx.df_metric == "coulomb":
        rho_KS_rec = rho_KS_rec[1:]
    rho_KS_rec = rho_KS_rec[sort_idx]

    partial_wave_coefs_prim_re, partial_wave_coefs_prim_im = gto_rec_prim(
        ctx.lmax_numba, ctx.species, ctx.npgf, ctx.alphas, Gvec_half, gcuts
    )
    pwc_g0, pwc_spread = gto_rec_g0(
        natoms, atomic_symbols, ctx.lmax, ctx.nmax_numba, ctx.npgf, ctx.alphas, ctx.contranorm, ncoefs
    )

    wp = get_w_prim(
        Gvec_half, natoms, atomic_coords, ctx.npgf, ctx.lmax_numba, atomic_symbols,
        partial_wave_coefs_prim_re, partial_wave_coefs_prim_im, volume, rho_KS_rec, ctx.df_metric, gcuts, rank,
    )
    C = build_contraction_matrix(natoms, atomic_symbols, ctx.lmax, ctx.nmax_numba, ctx.npgf, ctx.contranorm)
    w = C.T @ wp

    if ctx.df_metric == "identity":
        S = overlap_identity(cell, atomic_coords, atomic_symbols, ctx.nbasis, ncoefs, volume, ctx.pyscf_data, ctx.rcut_pairs)
    else:
        S_SR = overlap_coulomb_real(
            cell, atomic_coords, atomic_symbols, ctx.nbasis, ncoefs, volume, ctx.pyscf_data,
            ctx.pyscf_ewald, pwc_g0, pwc_spread, ctx.sigma_ewald, ctx.rcut_pairs,
        )
        partial_wave_coefs = gto_rec(
            ctx.lmax_numba, ctx.nmax_numba, ctx.nbasis, ctx.species, ctx.npgf, ctx.contranorm,
            ctx.alphas, Gvec_half, gmax_ewald_idx,
        )
        partial_wave_coefs_ewald = gto_rec_ewald(
            ctx.lmax_numba, ctx.lmax_max, ctx.nmax_numba, ctx.nbasis, ctx.species, ctx.npgf,
            ctx.contranorm, ctx.alphas, Gvec_half, gmax_ewald_idx, ctx.sigma_ewald,
        )
        S_LR = overlap_coulomb_rec(
            Gvec_half, natoms, atomic_coords, ctx.nbasis, ncoefs, atomic_symbols,
            partial_wave_coefs, partial_wave_coefs_ewald, volume, gmax_ewald_idx, rank,
        )
        S = S_SR + S_LR

    q = 4.0 * np.pi * pwc_g0
    Sinv = np.linalg.solve(S, np.column_stack([w, q]))
    Sinv_w, Sinv_q = Sinv[:, 0], Sinv[:, 1]
    lagmult = (np.dot(q, Sinv_w) - n_elec) / np.dot(q, Sinv_q)  # total-charge constraint
    c = Sinv_w - lagmult * Sinv_q

    return c, S


def main(geom_indexes: list[int] | None):
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"
    df_metric = inp.qm.dfmetric
    assert df_metric in ("identity", "coulomb"), f"{inp.qm.dfmetric=}, expected 'identity' or 'coulomb'"

    comm, size, rank, parallel = detect_mpi()
    if rank == 0:
        os.makedirs(osp.join(inp.salted.saltedpath, "coefficients"), exist_ok=True)
        os.makedirs(osp.join(inp.salted.saltedpath, "overlaps"), exist_ok=True)

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()

    if geom_indexes is None:
        geom_indexes = list(range(ndata))
    else:
        geom_indexes = [i for i in geom_indexes if i < ndata]  # indexes start from 0

    if parallel:
        check_MPI_tasks_count(comm, len(geom_indexes), "configurations")
    my_indexes = distribute_jobs(comm, geom_indexes)

    bdir = osp.join(inp.salted.saltedpath, "basis")
    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(
        lmax, nmax, species, inp.qm.dfbasis, bdir
    )
    pseudocharge = read_pseudocharge(species, inp)

    pyscf_data = setup_pyscf_species(species, lmax, nmax_numba, alphas, contranorm)
    pyscf_ewald = None
    sigma_ewald = None
    if df_metric == "identity":
        rcut_pairs = pair_cutoffs(species, lmax, alphas)
    else:
        sigma_ewald = 2.0 / bohr2angs  # 2 angstrom, hard-coded (same as salted.cp2k.density_fitting)
        pyscf_ewald = setup_pyscf_ewald(species, lmax, nmax_numba, alphas, contranorm, sigma_ewald)
        rcut_pairs = {(s1, s2): 4 * (sigma_ewald + sigma_ewald) for s1 in species for s2 in species}

    ctx = FitContext(
        species=species, lmax=lmax, nmax=nmax, lmax_max=lmax_max,
        atomic_symbols=atomic_symbols, atomic_coords=atomic_coords, natoms=natoms,
        df_metric=df_metric, path2qm=inp.qm.path2qm,
        alphas=alphas, contranorm=contranorm, npgf=npgf,
        lmax_numba=lmax_numba, nmax_numba=nmax_numba, nbasis=nbasis,
        pyscf_data=pyscf_data, pyscf_ewald=pyscf_ewald, rcut_pairs=rcut_pairs,
        sigma_ewald=sigma_ewald, pseudocharge=pseudocharge, verbose=inp.salted.verbose,
    )

    time_start = time.time()
    for geom_idx in my_indexes:
        if ctx.verbose:
            print(f"[rank {rank}] conf: {geom_idx + 1}", flush=True)
        try:
            c, S = fit_one(ctx, geom_idx, rank)
            np.save(osp.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{geom_idx}.npy"), c)
            np.save(osp.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{geom_idx}.npy"), S)
        except Exception as exc:
            print(f"[rank {rank}] FAILED conf_{geom_idx + 1}: {exc}", flush=True)

    if ctx.verbose:
        print(f"[rank {rank}] Total density fitting time: {time.time() - time_start:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=str, default="all", help=ARGHELP_INDEX_STR)
    args = parser.parse_args()

    main(parse_index_str(args.idx))
