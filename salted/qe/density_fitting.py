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
are intentionally not ported (they rely on GTH's analytic Gaussian ion charge). Instead, with the
coulomb metric, every configuration also reports the QE analogue of that diagnostic -- the
electrostatic energy E_ee + E_en + E_nn of the fitted density (`electrostatic_energies` below) --
computed in reciprocal space from quantities the fit already has in memory, so it needs neither
`salted.qe.df2cube` nor a reconstructed cube. See `salted.qe.electrostatic_energy` for the method
and its caveats (local pseudopotential only, G=0 dropped).

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
from salted.qe.electrostatic_energy import (
    ESPRESSO_PWO_FNAME,
    EWALD_RE,
    HARTREE_RE,
    compute_vloc_of_g,
    parse_upf_local,
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


def read_upf_local(species: list[str], inp) -> dict:
    """Each species' local pseudopotential (r_mesh, v_loc_ry, z_valence) from its UPF file, for
    the coulomb-metric electrostatic-energy report."""
    pseudo_map = inp.qm.pseudopotentials.to_dict()
    return {spe: parse_upf_local(osp.join(inp.qm.pseudo_dir, pseudo_map[spe])) for spe in species}


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


def _basis_index_labels(ctx: "FitContext", geom_idx: int) -> list[tuple[int, str, int, int, int]]:
    """(atom_idx, species, n, l, m) label for every row/column of this configuration's S, in the
    same atom-major, l-then-n-then-m order salted.cp2k.utils's basis-indexing functions use (see
    e.g. salted.qe.df2cube's identical iaux loop, which this mirrors).
    """
    atomic_symbols = ctx.atomic_symbols[geom_idx]
    natoms = ctx.natoms[geom_idx]
    labels = []
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        for l in range(ctx.lmax[spe] + 1):
            for n in range(ctx.nmax[(spe, l)]):
                for m in range(-l, l + 1):
                    labels.append((iat, spe, n, l, m))
    return labels


def diagnose_overlap_conditioning(S: np.ndarray, labels: list[tuple[int, str, int, int, int]],
                                   n_report: int = 5, rel_threshold: float = 1e-8) -> float:
    """Report cond(S) and which specific (atom, species, n, l, m) basis functions dominate S's
    smallest eigenvectors -- i.e. which functions are responsible for S being singular or
    ill-conditioned, so cutoffs/thresholds can be tailored to the functions actually at fault
    instead of by trial and error. Returns cond(S).

    S's singularity (`np.linalg.solve` raising `LinAlgError: Singular matrix`) is an exact test on
    whether LU factorization hits a zero pivot -- there's no single tunable "cutoff" number to
    compare against in advance. cond(S) (ratio of largest to smallest eigenvalue; S is symmetric
    positive (semi-)definite here) is the practical, continuous proxy: cond(S) approaching ~1e14-1e16
    (double precision's own limit) is functionally singular, but coefficients are already
    unreliable well before that, typically once cond(S) exceeds ~1e8-1e10.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    order = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    max_eig = eigenvalues[-1]
    cond = float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0 else float("inf")

    n_bad = int(np.sum(eigenvalues < rel_threshold * max_eig))
    print(f"    cond(S) = {cond:.3e}  (eigenvalues span {eigenvalues[0]:.3e} .. {eigenvalues[-1]:.3e})")
    print(f"    {n_bad} eigenvalue(s) below {rel_threshold:.0e} x largest -- these are what make S singular/ill-conditioned")

    for k in range(min(n_report, len(eigenvalues))):
        vec = eigenvectors[:, k]
        top = np.argsort(np.abs(vec))[::-1][:5]
        contributors = ", ".join(f"(atom {labels[i][0]} {labels[i][1]} n={labels[i][2]} l={labels[i][3]} m={labels[i][4]}):{vec[i]:+.2f}" for i in top)
        print(f"    eigenvalue[{k}] = {eigenvalues[k]:.3e} (ratio {eigenvalues[k] / max_eig:.1e} of largest) -- dominated by {contributors}")

    return cond


def ion_local_potential_rec(upf_local: dict, atomic_symbols, atomic_coords, Gvec_half, knorm_vec, volume) -> np.ndarray:
    """Fourier coefficients V_ion(G) [Ha] of the summed local pseudopotentials on `Gvec_half`
    (QE convention V(r) = sum_G V_ion(G) e^{iG.r}, i.e. V_ion(G) = (1/Omega) int V(r) e^{-iG.r} dr),
    using `salted.qe.electrostatic_energy.compute_vloc_of_g` (port of QE's vloc_of_g.f90).

    vloc(|G|) only depends on |G|, so it is evaluated once per unique |G| shell rather than once
    per G-vector -- on a regular grid that is far fewer radial quadratures.
    """
    knorm_unique, inverse = np.unique(np.round(knorm_vec, 10), return_inverse=True)
    v_ion_rec = np.zeros(len(Gvec_half), dtype=np.complex128)
    for spe in sorted(set(atomic_symbols)):
        r_mesh, v_loc_ry, z_valence = upf_local[spe]
        vloc_g = compute_vloc_of_g(r_mesh, v_loc_ry, z_valence, knorm_unique, volume)[inverse]
        structure_factor = np.zeros(len(Gvec_half), dtype=np.complex128)
        for iat, s in enumerate(atomic_symbols):
            if s == spe:
                structure_factor += np.exp(-1j * (Gvec_half @ atomic_coords[iat]))
        v_ion_rec += vloc_g * structure_factor
    return v_ion_rec


def read_qe_energies(conf_dir: str) -> tuple[float | None, float | None]:
    """(E_nn, E_hartree) in Ha from QE's `espresso.pwo` ("ewald contribution" / "hartree
    contribution"), or None for whichever is unavailable."""
    pwo_path = osp.join(conf_dir, ESPRESSO_PWO_FNAME)
    if not osp.exists(pwo_path):
        return None, None
    with open(pwo_path) as f:
        text = f.read()
    m_ewald, m_hartree = EWALD_RE.search(text), HARTREE_RE.search(text)
    e_nn = float(m_ewald.group(1)) / 2.0 if m_ewald else None  # Ry -> Ha
    e_hartree = float(m_hartree.group(1)) / 2.0 if m_hartree else None
    return e_nn, e_hartree


def electrostatic_energies(ctx: "FitContext", conf_dir: str, c, S, C, Gvec_half, knorm_vec, volume, natoms,
                           atomic_symbols, atomic_coords, pwc_prim_re, pwc_prim_im, gcuts, rho_KS_rec, rank) -> dict:
    """E_ee, E_en, E_nn (Ha) of the fitted density, coulomb metric only, all from in-memory fit data.

      E_ee = 1/2 c^T S c  (S is the coulomb-metric overlap, G=0 excluded)
      E_en = sum_i c_i <Phi_i|V_ion> = c . w_V
      E_nn = QE's own "ewald contribution"

    w_V reuses `get_w_prim` (the same projection that builds the fit's w = (Phi|rho_KS)_coulomb):
    the coulomb projection of a density rho_V equals <Phi|v_H[rho_V]>, with v_H(G) = 4*pi*rho_V(G)/G^2,
    so passing rho_V(G) = G^2 * Omega * V_ion(G) / (4*pi) (Omega converts V_ion's Fourier coefficients
    to the FFT*dV convention of rho_KS_rec) yields exactly <Phi_i|V_ion>. This is equivalent to
    `salted.qe.electrostatic_energy`'s cube-based E_en, but with the basis functions' exact Fourier
    transforms instead of a real-space reconstruction truncated at rcut.

    Also returns E_en and the total for QE's true density (rho_KS, the fit's target) for comparison.
    """
    v_ion_rec = ion_local_potential_rec(ctx.upf_local, atomic_symbols, atomic_coords, Gvec_half, knorm_vec, volume)

    rho_v_rec = knorm_vec**2 * volume * v_ion_rec / (4.0 * np.pi)
    wp_v = get_w_prim(
        Gvec_half, natoms, atomic_coords, ctx.npgf, ctx.lmax_numba, atomic_symbols,
        pwc_prim_re, pwc_prim_im, volume, rho_v_rec, "coulomb", gcuts, rank,
    )
    e_en = float(c @ (C.T @ wp_v))
    e_ee = 0.5 * float(c @ (S @ c))
    e_en_true = 2.0 * float(np.sum((np.conj(rho_KS_rec) * v_ion_rec).real))  # half-sphere, G != 0

    e_nn, e_hartree_qe = read_qe_energies(conf_dir)
    result = {"e_ee": e_ee, "e_en": e_en, "e_nn": e_nn, "e_en_true": e_en_true, "e_hartree_qe": e_hartree_qe,
              "e_total": None, "e_total_true": None}
    if e_nn is not None:
        result["e_total"] = e_ee + e_en + e_nn
        if e_hartree_qe is not None:
            result["e_total_true"] = e_hartree_qe + e_en_true + e_nn
    return result


def format_electrostatics(conf_n: int, en: dict, verbose: bool) -> str:
    """Per-configuration report, mirroring salted.cp2k.density_fitting's Hartree-energy line."""
    def fmt(x):
        return f"{x:.8f}" if x is not None else "n/a"

    lines = [f"conf {conf_n}: electrostatic energy E_ee+E_en+E_nn = {fmt(en['e_total'])} Ha "
             f"(E_ee = {fmt(en['e_ee'])}, E_en = {fmt(en['e_en'])}, E_nn = {fmt(en['e_nn'])} Ha)"]
    if en["e_nn"] is None:
        lines.append(f"conf {conf_n}:   no QE ewald contribution found in {ESPRESSO_PWO_FNAME}; total not available")
    if verbose:
        lines.append(f"conf {conf_n}:   true QE density: E_en = {fmt(en['e_en_true'])} Ha (fit - true: {en['e_en'] - en['e_en_true']:+.6f}), "
                     f"E_ee(QE hartree) = {fmt(en['e_hartree_qe'])} Ha, total = {fmt(en['e_total_true'])} Ha")
        if en["e_total"] is not None and en["e_total_true"] is not None:
            lines[-1] += f" (fit - true: {en['e_total'] - en['e_total_true']:+.6f})"
    return "\n".join(lines)


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
    upf_local: dict | None = None  # species -> (r_mesh, v_loc_ry, z_valence); coulomb metric only


def fit_one(ctx: FitContext, geom_idx: int, rank: int) -> tuple[np.ndarray, np.ndarray, dict | None]:
    """Fit a single configuration's density onto the RI basis. Returns (coefficients, overlap,
    electrostatic energies) -- the latter only for the coulomb metric, else None (also None, with a
    warning, if the energy evaluation itself fails: that must never discard a successful fit)."""
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
    try:
        Sinv = np.linalg.solve(S, np.column_stack([w, q]))
    except np.linalg.LinAlgError as exc:
        print(f"[conf {geom_idx + 1}] overlap matrix S is singular ({exc}); diagnosing which basis functions are at fault:", flush=True)
        diagnose_overlap_conditioning(S, _basis_index_labels(ctx, geom_idx))
        raise
    if ctx.verbose:
        cond = np.linalg.cond(S)
        warning = " -- already large enough for coefficients to be unreliable" if cond > 1e8 else ""
        print(f"[conf {geom_idx + 1}] cond(S) = {cond:.3e}{warning}", flush=True)
    Sinv_w, Sinv_q = Sinv[:, 0], Sinv[:, 1]
    lagmult = (np.dot(q, Sinv_w) - n_elec) / np.dot(q, Sinv_q)  # total-charge constraint
    c = Sinv_w - lagmult * Sinv_q

    energies = None
    if ctx.df_metric == "coulomb" and ctx.upf_local is not None:
        time_e = time.time()
        try:
            energies = electrostatic_energies(
                ctx, conf_dir, c, S, C, Gvec_half, knorm_vec, volume, natoms, atomic_symbols, atomic_coords,
                partial_wave_coefs_prim_re, partial_wave_coefs_prim_im, gcuts, rho_KS_rec, rank,
            )
        except Exception as exc:
            print(f"[conf {geom_idx + 1}] WARNING: electrostatic energy evaluation failed ({exc}); fit is unaffected", flush=True)
        if ctx.verbose:
            print(f"[conf {geom_idx + 1}] time to compute electrostatic energy: {time.time() - time_e:.2f}s", flush=True)

    return c, S, energies


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
        upf_local=read_upf_local(species, inp) if df_metric == "coulomb" else None,
    )

    time_start = time.time()
    for geom_idx in my_indexes:
        if ctx.verbose:
            print(f"[rank {rank}] conf: {geom_idx + 1}", flush=True)
        try:
            c, S, energies = fit_one(ctx, geom_idx, rank)
            np.save(osp.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{geom_idx}.npy"), c)
            np.save(osp.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{geom_idx}.npy"), S)
            if energies is not None:
                print(format_electrostatics(geom_idx + 1, energies, ctx.verbose), flush=True)
        except Exception as exc:
            print(f"[rank {rank}] FAILED conf_{geom_idx + 1}: {exc}", flush=True)

    if ctx.verbose:
        print(f"[rank {rank}] Total density fitting time: {time.time() - time_start:.2f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=str, default="all", help=ARGHELP_INDEX_STR)
    args = parser.parse_args()

    main(parse_index_str(args.idx))
