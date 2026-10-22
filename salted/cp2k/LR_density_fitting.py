"""Long-range (LR) density fitting of CP2K electron densities

Same workflow as salted.cp2k.density_fitting, but only the long-range part of the density is
fitted: the density is smoothed with a normalised Gaussian of width sigma = 1 A (SIGMA_LR),

    rho_LR(G) = exp(-sigma^2 G^2 / 2) rho(G),

so that U_LR = 1/2 (rho_tot,LR | rho_tot,LR) is the Ewald long-range energy with kernel
erf(r / 2 sigma) / r. The fit is done fully in reciprocal space on a small basis of
primitive Gaussians whose widths lie in the band [sigma, sqrt(sigma^2 + s_max^2)] of the
CP2K RI basis (qm.dfbasis). Two basis recipes (lr.recipe in inp.yaml):
    minimal (default): L <= 1 for every species, 3 widths for l = 0 and 1 for l = 1
    accurate:          L <= 2 with 3/2/1 widths, except H, which keeps L <= 1 with 3/1
Differences with density_fitting.py are marked with "LR:".

Usage (conf_start and conf_end are 0-based, as in density_fitting.py):

    python -m salted.cp2k.LR_density_fitting conf_start conf_end [--cond]

with --cond to also compute the condition number of the metric matrix (an eigendecomposition).
The LR settings are in the lr section of inp.yaml. The outputs, in <saltedpath>, are named so
that those of density_fitting.py are left untouched:
    coefficients_lr/coefficients_conf<i>.npy, overlaps_lr/overlap_conf<i>.npy
    density_lr/density_conf<i>.npz      LR density on a coarse grid (input for lr.from_density)
    metrics_lr/metrics_conf<i>.json     error metrics, with a table in metrics_lr/summary.dat
The LR basis is written to <saltedpath>/basis/ and registered in <saltedpath>/basis/lr_basis.yaml.
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np
from ase.data import atomic_numbers

from salted.basis_client import BasisClient
from salted.constants import bohr2angs
from salted.cp2k.utils import (
    SIGMA_LR,
    build_gvec,
    get_basis_set_info_numba,
    gto_rec,
    gto_rec_g0,
    lr_basis_settings,
    overlap_coulomb_rho,
    read_local_pseudo,
)
from salted.sys_utils import (
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    read_system,
    save_overlap,
)

HARTREE2MEV = 27211.386245988

def lr_basis_widths(lmax, nrad, sigma, rloc, smax):
    """Widths w (bohr) of the LR primitives r^l exp(-r^2 / 2 w^2) of one species, per l,
    from tight to diffuse."""
    widths = {}
    for l in range(lmax + 1):
        lo = np.sqrt(sigma**2 + rloc**2) if l == 0 else sigma # sqrt(sigma^2 + r_loc^2) for l = 0 to match the smoothed core charge, else sigma
        hi = max(np.sqrt(sigma**2 + smax.get(l, max(smax.values())) ** 2), 1.05 * lo) # sqrt(sigma^2 + s_max,l^2) or 1.05 * lo if the band is too narrow
        if nrad[l] == 1:
            widths[l] = np.array([lo if l == 0 else np.sqrt(lo * hi)]) # tightest for l = 0, else geometric mean of the band
        else:
            widths[l] = np.geomspace(lo, hi, nrad[l]) # spread functions evenly in log space, from tight to diffuse
    return widths


def write_lr_basis(bdir, dfbasis_lr, widths, yaml_path):
    """Write the LR basis in the format of salted.cp2k.get_basis_info: alphas in decreasing
    order (as in CP2K, so the last l = 0 function is the most diffuse one) and an identity contraction.
    Registers the basis in yaml_path and returns lmax, nmax."""
    basis_data = {}
    for spe in widths:
        for l in widths[spe]:
            alphas = np.sort(1.0 / (2.0 * widths[spe][l] ** 2))[::-1]
            np.savetxt(osp.join(bdir, f"{spe}-{dfbasis_lr}-alphas-L{l}.dat"), alphas)
            np.savetxt(osp.join(bdir, f"{spe}-{dfbasis_lr}-contra-L{l}.dat"), np.eye(len(alphas)))
        basis_data[spe] = {"lmax": max(widths[spe]), "nmax": [len(widths[spe][l]) for l in sorted(widths[spe])]}
    if not osp.exists(yaml_path):
        open(yaml_path, "w").close()
    BasisClient(data_fpath=yaml_path).write_to_yaml(dfbasis_lr, basis_data, force_overwrite=True)


def metric_weight_lr(knorm_vec, df_metric):
    if df_metric == "coulomb":
        weight = 2.0 * 4.0 * np.pi / knorm_vec**2
    elif df_metric == "identity":
        weight = np.full(len(knorm_vec), 2.0)
        weight[knorm_vec == 0.0] = 1.0
    return weight


def w_lr_rec(Gvec_half, knorm_vec, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, volume, rho_rec, df_metric, chunk=2048):
    """Density projections w_i = <Phi_i|O|rho> in reciprocal space
    (the LR basis is made of normalised primitives: no contraction and no per-primitive G-cutoffs are needed)"""
    w = np.zeros(ncoefs)
    rho_w = metric_weight_lr(knorm_vec, df_metric) * rho_rec

    for g0 in range(0, len(knorm_vec), chunk):
        g1 = min(g0 + chunk, len(knorm_vec))
        A = basis_rec(Gvec_half[g0:g1], natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, g0, g1)
        w += A.real.T @ rho_w[g0:g1].real + A.imag.T @ rho_w[g0:g1].imag

    return w / volume


def overlap_lr_rec(Gvec_half, knorm_vec, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, volume, df_metric, chunk=2048):
    # Metric matrix S_ij = <Phi_i|O|Phi_j> in reciprocal space
    S = np.zeros((ncoefs, ncoefs))
    weight = metric_weight_lr(knorm_vec, df_metric)

    for g0 in range(0, len(knorm_vec), chunk):
        g1 = min(g0 + chunk, len(knorm_vec))
        # Basis functions in reciprocal space: 4 pi * pwc_i(G) * e^{-iG.r_iat}, scaled by sqrt(weight)
        A = basis_rec(Gvec_half[g0:g1], natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, g0, g1)
        A *= np.sqrt(weight[g0:g1])[:, np.newaxis]
        A_real = np.ascontiguousarray(A.real)
        A_imag = np.ascontiguousarray(A.imag)
        S += A_real.T @ A_real + A_imag.T @ A_imag

    return S / volume


def basis_rec(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, g0, g1):
    # Fourier components 4 pi * pwc_i(G) * e^{-iG.r_iat} of all basis functions, for the G-vectors g0:g1
    A = np.empty((g1 - g0, ncoefs), dtype=np.complex128)
    phase = np.exp(-1j * np.dot(Gvec_half, coords.T)) # e^{-iG.r_iat}
    icoefs = 0
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        A[:, icoefs:icoefs + nbasis[spe]] = 4.0 * np.pi * partial_wave_coefs[spe][g0:g1] * phase[:, iat, np.newaxis]
        icoefs += nbasis[spe]
    return A


def rho_fit_rec(Gvec_half, natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, coefs, chunk=2048):
    # Fourier components of the fitted density sum_i c_i Phi_i
    rho = np.empty(len(Gvec_half), dtype=np.complex128)
    for g0 in range(0, len(Gvec_half), chunk):
        g1 = min(g0 + chunk, len(Gvec_half))
        rho[g0:g1] = basis_rec(Gvec_half[g0:g1], natoms, coords, nbasis, ncoefs, atomic_symbols, partial_wave_coefs, g0, g1) @ coefs
    return rho


def get_rho_n_rec_lr(Gvec_half, knorm_vec, natoms, coords, atomic_symbols, pseudocharge, rloc, sigma):
    # Fourier components of the Gaussian core charges (as get_rho_n, width rloc) smoothed by the LR filter,
    # computed analytically: sum_iat Z exp(-(rloc^2 + sigma^2) G^2 / 2) e^{-iG.r_iat}
    rho_n_rec = np.zeros(len(knorm_vec), dtype=np.complex128)
    phase = np.exp(-1j * np.dot(Gvec_half, coords.T))
    for iat in range(natoms):
        spe = atomic_symbols[iat]
        rho_n_rec += pseudocharge[spe] * np.exp(-0.5 * (rloc[spe]**2 + sigma**2) * knorm_vec**2) * phase[:, iat]
    return rho_n_rec


def odd_fft_size(nmin):
    # Smallest odd n >= nmin with prime factors 3, 5, 7 only
    n = max(3, int(nmin) | 1)
    while True:
        k = n
        for p in (3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return n
        n += 2


def lr_grid_size(Gvec_half, cell_lengths, spacing=None):
    # Odd real-space grid that holds all the G-vectors exactly (optionally with a maximum spacing, in bohr)
    mmax = np.rint(np.abs(Gvec_half) * cell_lengths / (2.0 * np.pi)).astype(int).max(axis=0)
    nmin = 2 * mmax + 1
    if spacing is not None:
        nmin = np.maximum(nmin, np.ceil(cell_lengths / spacing).astype(int))
    return tuple(odd_fft_size(n) for n in nmin)


def rec_to_grid(rho_rec, Gvec_half, cell_lengths, origin, nside, rho_g0=None):
    # Values on the grid r_j = origin + j * L / nside of the band-limited function with half-space Fourier
    m = np.rint(Gvec_half * cell_lengths / (2.0 * np.pi)).astype(int)
    rho_fft = np.zeros((nside[0], nside[1], nside[2] // 2 + 1), dtype=np.complex128)
    vals = rho_rec * np.exp(1j * np.dot(Gvec_half, origin))
    rho_fft[m[:, 0] % nside[0], m[:, 1] % nside[1], m[:, 2]] = vals
    plane = m[:, 2] == 0 # irfftn needs both G and -G in the Gz = 0 plane
    rho_fft[(-m[plane, 0]) % nside[0], (-m[plane, 1]) % nside[1], 0] = np.conj(vals[plane])
    if rho_g0 is not None:
        rho_fft[0, 0, 0] = rho_g0
    return np.fft.irfftn(rho_fft, s=nside, axes=(0, 1, 2)) * np.prod(nside) / np.prod(cell_lengths)


def write_cube(path, title, rho, cell_lengths, origin, valences, pseudocharges, coords):
    # Cube file of an orthorhombic grid, 6 values per line
    nside = rho.shape
    with open(path, "w") as cubef:
        print(title, file=cubef)
        print("CUBE FORMAT", file=cubef)
        print(len(valences), origin[0], origin[1], origin[2], file=cubef)
        metric = np.diag(np.asarray(cell_lengths) / np.asarray(nside))
        for k in range(3):
            print(nside[k], metric[k, 0], metric[k, 1], metric[k, 2], file=cubef)
        for iat in range(len(valences)):
            print(valences[iat], pseudocharges[iat], coords[iat][0], coords[iat][1], coords[iat][2], file=cubef)
        for row in rho.reshape(-1, nside[2]):
            for i0 in range(0, len(row), 6):
                print(" ".join(f"{x:13.5E}" for x in row[i0:i0 + 6]), file=cubef)


def write_summary(path, metrics):
    # Table of the per-configuration metrics
    with open(path, "w") as f:
        f.write("#  conf natoms  ncoefs       nG     U_LR_ref(Ha)     U_LR_fit(Ha) dU(meV/at) -1/2(D|D)(meV/at)"
                "   MAE(%)     cond(S)  N_grid-N_elec  time(s)\n")
        f.writelines(f"{r['conf']:6d} {r['natoms']:6d} {r['ncoefs']:7d} {r['nG']:8d} {r['e_hartree_ref']:16.8f} "
                    f"{r['e_hartree']:16.8f} {r['dU_meV_per_atom']:10.5f} {r['half_DD_meV_per_atom']:17.5f} "
                    f"{r['mae_percent']:8.4f} {r['cond_S']:11.2e} {r['n_elec_grid'] - r['n_elec']:+14.2e} {r['time']:8.1f}\n" for r in metrics)


if __name__ == "__main__":

    # LR: conf_start and conf_end as in density_fitting.py, plus --cond
    parser = argparse.ArgumentParser(description="Long-range density fitting of CP2K densities")
    parser.add_argument("conf_start", type=int, help="first configuration, 0-based")
    parser.add_argument("conf_end", type=int, help="last configuration, 0-based, included")
    parser.add_argument("--cond", action="store_true", help="also compute the condition number of S (an eigendecomposition)")
    args = parser.parse_args()

    inp = ParseConfig().parse_input()
    df_metric = inp.qm.dfmetric

    conf_start = args.conf_start
    conf_end = args.conf_end

    comm, size, rank, parallel = detect_mpi()

    if rank==0: print("Parallel run over", size, "tasks",flush=True)

    # LR: filter width and G-vector truncation
    sigma_lr = SIGMA_LR / bohr2angs # LR filter width (bohr)
    gmax_lr = 2.0 * np.pi / sigma_lr # G-vector truncation, filter exp(-2 pi^2) ~ 3e-9
    # LR: basis recipe (lr.recipe) with the per-species overrides of lr.lmax / lr.nrad
    dfbasis_lr, lmax_lr, nrad_lr = lr_basis_settings(inp)

    # LR: output folders
    coefpath = osp.join(inp.salted.saltedpath, "coefficients_lr")
    overpath = osp.join(inp.salted.saltedpath, "overlaps_lr")
    denspath = osp.join(inp.salted.saltedpath, "density_lr")
    metrpath = osp.join(inp.salted.saltedpath, "metrics_lr")

    if rank==0:

        for dirpath in [coefpath, overpath, metrpath] + ([denspath] if inp.lr.from_density is None else []):
            if not os.path.exists(dirpath):
                os.mkdir(dirpath)

    conf_list = np.arange(conf_start, conf_end + 1)

    if parallel:
        comm.Barrier()
        check_MPI_tasks_count(comm, len(conf_list), "configurations")
        conf_range = distribute_jobs(comm, conf_list)
        print(f"Task {rank+1} handles the following configurations: {conf_range}", flush=True)
    else:
        conf_range = conf_list

    # Initialize SALTED
    time_start = time.time()
    species, lmax_cp2k, nmax_cp2k, lmax_max, nnmax, ndata, atomic_symbols, atomic_coords, natoms, natmax = read_system()

    bdir = osp.join(inp.salted.saltedpath,"basis")
    pseudocharge, rloc = read_local_pseudo(species, bdir)

    # LR: define the LR basis from the recipe (minimal or accurate) and the width band of the CP2K basis,
    # then read it back as in density_fitting.py
    widths = {}
    for spe in species:
        smax = {} # most diffuse CP2K width at each l
        for l in range(lmax_cp2k[spe]+1):
            smax[l] = 1.0 / np.sqrt(2.0 * np.loadtxt(osp.join(bdir, f"{spe}-{inp.qm.dfbasis}-alphas-L{l}.dat")).min()) # smax = 1 / sqrt(2 alpha_min)
        widths[spe] = lr_basis_widths(lmax_lr[spe], nrad_lr[spe], sigma_lr, rloc[spe], smax)
    lmax = {spe: max(widths[spe]) for spe in species}
    nmax = {(spe, l): len(widths[spe][l]) for spe in species for l in widths[spe]}

    if rank==0:
        write_lr_basis(bdir, dfbasis_lr, widths, osp.join(bdir, "lr_basis.yaml"))
        print(f"LR basis {dfbasis_lr}: sigma = {SIGMA_LR} A, G_max = {gmax_lr:.3f} bohr^-1, metric {df_metric}", flush=True)
        for spe in species:
            print(f"  {spe}: " + "; ".join(f"l={l}: " + ", ".join(f"{w * bohr2angs:.3f}" for w in widths[spe][l])
                                         for l in widths[spe]) + " A", flush=True)
    if parallel:
        comm.Barrier()

    lmax_numba, nmax_numba, npgf, nbasis, alphas, contranorm = get_basis_set_info_numba(lmax, nmax, species, dfbasis_lr, bdir)

    for iconf in conf_range:

        time_conf = time.time()
        if inp.salted.verbose: print("conf:", iconf+1)

        # Compute coefs array size and number of electrons
        ntype = {}
        n_elec = 0.0
        for spe in species:
            ntype[spe] = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            ntype[spe] += 1
            n_elec += pseudocharge[spe]
        ncoefs = 0
        for spe in species:
            ncoefs += nbasis[spe]*ntype[spe]

        if inp.lr.from_density is None:

            # Read in electron density from cube files
            cubefile_pattern = os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", "*ELECTRON_DENSITY-1_0.cube")
            with open(glob.glob(cubefile_pattern)[0], "r") as cubefile:
                # Header: 2 comment lines + origin line + 3 lattice lines + atom lines
                header = [cubefile.readline() for _ in range(6 + natoms[iconf])]
                line_ori, line_x, line_y, line_z = (header[i].split() for i in range(2, 6))
                nx = int(line_x[0])
                ny = int(line_y[0])
                nz = int(line_z[0])
                dx = float(line_x[1])
                dy = float(line_y[2])
                dz = float(line_z[3])
                origin = np.asarray(line_ori[1:4], dtype=float)
                npoints = nx * ny * nz
                if inp.salted.verbose: print("Number of grid points:", npoints)
                # LR: in vacuum regions CP2K writes 3-digit exponents and negative values fill their 13-character field,
                # so they run into the previous value ("0.53102E-017-0.16205E-016"), mantissas are always 0.xxxxx and
                # exponents never contain "-0.", so a space before each "-0." separates them
                rho_KS = np.fromstring(cubefile.read().replace("-0.", " -0."), dtype=np.float64, sep=' ')

        else:

            # LR: read a stored LR density (coarse grid, already filtered), written by a previous run in density_lr/
            lrdata = np.load(osp.join(inp.lr.from_density, f"density_conf{iconf}.npz"))
            rho_KS = lrdata["rho_lr"]
            nx, ny, nz = rho_KS.shape
            dx, dy, dz = lrdata["cell_lengths"] / np.array([nx, ny, nz])
            origin = lrdata["origin"]
            npoints = nx * ny * nz
            if abs(float(lrdata["sigma"]) - sigma_lr) > 1e-10:
                raise ValueError(f"stored LR density of conf {iconf+1} filtered with sigma = {float(lrdata['sigma']) * bohr2angs} A, "
                                 f"not {SIGMA_LR} A")

        dr = np.array([dx, dy, dz])

        if rho_KS.size != npoints:
            print("ERROR: inconsistent number of grid points!")
            sys.exit(0)

        rho_KS = rho_KS.reshape((nx, ny, nz))
        volume = (nx*dx)*(ny*dy)*(nz*dz)
        n_elec_grid = np.sum(rho_KS) * np.prod(dr) # G = 0 component of the density

        # Get cell
        cell = np.zeros((3,3))
        cell[0,0] = nx*dx
        cell[1,1] = ny*dy
        cell[2,2] = nz*dz
        cell_lengths = np.diag(cell)

        # Generate sorted G-vectors for the half-space Z>0
        Gvec_half, knorm_vec, gidx = build_gvec((nx, ny, nz), dr, df_metric)

        # Compute density Fourier-components
        rho_KS_rec = np.fft.fftn(rho_KS).ravel()[gidx] * np.prod(dr)
        del rho_KS

        # LR: keep the G-vectors with |G| < 2 pi / sigma and apply the Gaussian filter, the phase accounts for the cube origin
        nG = np.searchsorted(knorm_vec, gmax_lr).astype(np.int64) # Index of the last G-vector below the cutoff
        Gvec_half = np.ascontiguousarray(Gvec_half[:nG])
        knorm_vec = knorm_vec[:nG]
        rho_KS_rec = rho_KS_rec[:nG] * np.exp(-1j * np.dot(Gvec_half, origin))
        if inp.lr.from_density is None:
            rho_KS_rec *= np.exp(-0.5 * sigma_lr**2 * knorm_vec**2) # Apply the Gaussian filter to the density (stored densities are already filtered)
        if inp.salted.verbose: print("Number of LR G-vectors:", nG)

        # LR: store the LR density on the coarsest exact grid
        if inp.lr.from_density is None:
            nside_lr = lr_grid_size(Gvec_half, cell_lengths)
            rho_lr = rec_to_grid(rho_KS_rec, Gvec_half, cell_lengths, origin, nside_lr, rho_g0=n_elec_grid)
            np.savez_compressed(osp.join(denspath, f"density_conf{iconf}.npz"),
                                rho_lr=rho_lr, cell_lengths=cell_lengths, origin=origin, sigma=sigma_lr)
            if inp.lr.write_cube:
                valences = [atomic_numbers[spe] for spe in atomic_symbols[iconf]]
                write_cube(osp.join(denspath, f"density_conf{iconf}.cube"), f"LR electron density, sigma = {SIGMA_LR} A",
                           rho_lr, cell_lengths, origin, valences, [pseudocharge[spe] for spe in atomic_symbols[iconf]], atomic_coords[iconf])
            if inp.salted.verbose: print("LR density stored on a grid of", nside_lr, "points")

        # Compute partial-wave coefs as basis set fourier transform
        # LR: the LR basis functions are normalised primitives
        time_c = time.time()
        partial_wave_coefs = gto_rec(lmax_numba, nmax_numba, nbasis, species, npgf, contranorm, alphas, Gvec_half, nG)
        pwc_g0, pwc_spread = gto_rec_g0(natoms[iconf], atomic_symbols[iconf], lmax, nmax_numba, npgf, alphas, contranorm, ncoefs) # Monopole and spread
        if inp.salted.verbose: print("Time to compute pw coeffs:", time.time()-time_c)

        # Compute density projections <Phi|O|rho> fully in reciprocal space
        # LR: directly on the LR basis
        time_c = time.time()
        w = w_lr_rec(Gvec_half, knorm_vec, natoms[iconf], atomic_coords[iconf], nbasis, ncoefs, atomic_symbols[iconf], partial_wave_coefs, volume, rho_KS_rec, df_metric)
        if inp.salted.verbose: print("Time to build w:", time.time()-time_c)

        # LR: compute the metric matrix S_ij = <Phi_i|O|Phi_j> in reciprocal space
        time_c = time.time()
        S = overlap_lr_rec(Gvec_half, knorm_vec, natoms[iconf], atomic_coords[iconf], nbasis, ncoefs, atomic_symbols[iconf], partial_wave_coefs, volume, df_metric)
        if inp.salted.verbose: print("Time to build S:", time.time()-time_c)

        # Solve density-fitting linear system
        time_c = time.time()
        q = 4.0 * np.pi * pwc_g0
        Sinv = np.linalg.solve(S, np.column_stack([w, q]))
        Sinv_w, Sinv_q = Sinv[:, 0], Sinv[:, 1]
        lagmult = (np.dot(q, Sinv_w) - n_elec) / np.dot(q, Sinv_q) # Lagrange multiplier
        c = Sinv_w - lagmult * Sinv_q # Fitted coefficients

        if inp.salted.verbose: print("Time to solve linear system:", time.time()-time_c)

        # Save data
        np.save(os.path.join(coefpath, f"coefficients_conf{iconf}.npy"), c)
        save_overlap(os.path.join(overpath, f"overlap_conf{iconf}.npy"), S) # lower triangle only

        # LR: fitted density in reciprocal space, for the error metrics
        rho_fit = rho_fit_rec(Gvec_half, natoms[iconf], atomic_coords[iconf], nbasis, ncoefs, atomic_symbols[iconf], partial_wave_coefs, c)
        delta_rec = rho_KS_rec - rho_fit

        # Hartree energy calculation
        metrics = {"conf": int(iconf), "dfbasis_lr": dfbasis_lr, "natoms": int(natoms[iconf]), "ncoefs": int(ncoefs), "nG": int(nG),
                   "n_elec": float(n_elec), "n_elec_grid": float(n_elec_grid), "lagmult": float(lagmult)}
        time_c = time.time()
        if df_metric == "coulomb":
            # LR: core charges smoothed analytically
            rho_n_rec = get_rho_n_rec_lr(Gvec_half, knorm_vec, natoms[iconf], atomic_coords[iconf], atomic_symbols[iconf], pseudocharge, rloc, sigma_lr)
            # LR: same terms as in compute_hartree_energy, with the projections on the LR basis
            e_ee = 0.5 * np.dot(c, S @ c) # Electron-electron term: E_ee = 1/2 c^T.S.c
            wn = w_lr_rec(Gvec_half, knorm_vec, natoms[iconf], atomic_coords[iconf], nbasis, ncoefs, atomic_symbols[iconf], partial_wave_coefs, volume, rho_n_rec, df_metric)
            e_en = -np.dot(c, wn) # Electron-nucleus term: E_en = -sum_i c_i (Phi_i|rho_n)
            e_nn = 0.5 * overlap_coulomb_rho(rho_n_rec, rho_n_rec, knorm_vec, volume) # Nucleus-nucleus term: E_nn = 1/2 (rho_n|rho_n)
            e_hartree = e_ee + e_en + e_nn
            # LR: reference LR energy, directly from the LR density (no fit)
            e_ee_ref = 0.5 * overlap_coulomb_rho(rho_KS_rec, rho_KS_rec, knorm_vec, volume)
            e_en_ref = -overlap_coulomb_rho(rho_n_rec, rho_KS_rec, knorm_vec, volume)
            e_nn_ref = e_nn
            e_hartree_ref = e_ee_ref + e_en_ref + e_nn_ref

            # LR: e_hartree - e_hartree_ref = -1/2 (D|D)
            delta_delta = overlap_coulomb_rho(delta_rec, delta_rec, knorm_vec, volume)
            print(f"conf {iconf+1}: LR Hartree energy = {e_hartree:.8f} Ha, reference = {e_hartree_ref:.8f} Ha, "
                  f"error = {(e_hartree - e_hartree_ref) * HARTREE2MEV / natoms[iconf]:+.5f} meV/atom "
                  f"(-1/2 (D|D) = {-0.5 * delta_delta * HARTREE2MEV / natoms[iconf]:+.5f} meV/atom)", flush=True)
            if inp.salted.verbose: print(f"conf {iconf+1}: E_ee = {e_ee:.8f} Ha, E_en = {e_en:.8f} Ha, E_nn = {e_nn:.8f} Ha", flush=True)
            if inp.salted.verbose: print('Time to compute Hartree energy:', time.time() - time_c, flush=True)
            metrics.update({"e_hartree": e_hartree, "e_hartree_ref": e_hartree_ref,
                            "e_ee_en_nn": [e_ee, e_en, e_nn], "e_ee_en_nn_ref": [e_ee_ref, e_en_ref, e_nn_ref],
                            "dU_meV_per_atom": (e_hartree - e_hartree_ref) * HARTREE2MEV / natoms[iconf],
                            "half_DD_meV_per_atom": -0.5 * delta_delta * HARTREE2MEV / natoms[iconf]})

        # LR: MAE of the fitted LR density on a grid with spacing <= sigma / 4 (the G = 0 error, N_grid - n_elec, is left out)
        time_c = time.time()
        nside_mae = lr_grid_size(Gvec_half, cell_lengths, spacing=0.25 * sigma_lr)
        delta = rec_to_grid(delta_rec, Gvec_half, cell_lengths, origin, nside_mae, rho_g0=0.0)
        mae = 100.0 * np.sum(np.abs(delta)) * volume / np.prod(nside_mae) / n_elec
        # LR: condition number of the normalised S (only with --cond)
        if args.cond:
            d = 1.0 / np.sqrt(np.diag(S))
            eigs = np.linalg.eigvalsh(d[:, np.newaxis] * S * d[np.newaxis, :])
            cond = eigs[-1] / eigs[0] if eigs[0] > 0 else float("inf")
        else:
            cond = float("nan")
        print(f"conf {iconf+1}: MAE of the LR density = {mae:.4f} %, cond(S) = {cond:.2e}, ncoefs = {ncoefs}, "
              f"|c| = {np.linalg.norm(c):.3e}, max|c| = {np.abs(c).max():.3e}", flush=True)
        if cond > 1e14:
            # LR: e.g. metal slabs with large bases: more functions than significant LR Fourier components
            print(f"WARNING conf {iconf+1}: S is numerically singular (cond = {cond:.1e}): the density and energy are fitted, "
                  "but the coefficients are not uniquely defined, consider a smaller LR basis", flush=True)
        if inp.salted.verbose: print('Time to compute the errors:', time.time() - time_c, flush=True)
        metrics.update({"mae_percent": float(mae), "cond_S": float(cond), "norm_c": float(np.linalg.norm(c)),
                        "max_abs_c": float(np.abs(c).max()), "time": time.time() - time_conf})
        with open(osp.join(metrpath, f"metrics_conf{iconf}.json"), "w") as f:
            json.dump(metrics, f, indent=1)

    # LR: table of the metrics of all the configurations fitted with this LR basis
    if parallel:
        comm.Barrier()
    if rank==0 and df_metric == "coulomb":
        metrics_all = [json.load(open(path)) for path in glob.glob(osp.join(metrpath, "metrics_conf*.json"))]
        write_summary(osp.join(metrpath, "summary.dat"),
                      sorted((m for m in metrics_all if m["dfbasis_lr"] == dfbasis_lr), key=lambda m: m["conf"]))

    time_end = time.time()
    if inp.salted.verbose: print("Total density fitting time:", time_end-time_start, flush=True)
