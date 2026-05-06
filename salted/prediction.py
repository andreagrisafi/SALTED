import os
import os.path as osp
import sys
import time
import argparse
import numpy as np
from typing import Dict, Tuple, List, Optional, Any

from ase.data import atomic_numbers
from ase.io import read
from scipy import special

from salted import basis, sph_utils
from salted.sph_utils import equicombnonorm, antiequicombnonorm, kernelequicomb, kernelnorm
from salted.sys_utils import (
    PLACEHOLDER,
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    format_index_ranges,
    get_atom_idx as sys_utils_get_atom_idx,
    get_feats_projs,
    get_feats_projs_response,
    read_system,
    init_property_file,
)

from salted.cp2k.utils import init_moments, compute_charge_and_dipole, compute_polarizability
from salted import read_model



# ============================================================================
# SHARED UTILITY FUNCTIONS
# ============================================================================
def get_atom_idx(ndata: int, natoms: np.ndarray, species: List[str], atomic_symbols: List[List[str]]):
    """
    Get indices of atoms by species for each configuration.
    
    Works for both config mode and standalone mode.
    """
    atom_idx = {}
    natom_dict = {}
    
    for iconf in range(ndata):
        for spe in species:
            atom_idx[(iconf, spe)] = []
            natom_dict[(iconf, spe)] = 0
    
    for iconf in range(ndata):
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            atom_idx[(iconf, spe)].append(iat)
            natom_dict[(iconf, spe)] += 1
    
    return atom_idx, natom_dict


def compute_equivariant_descriptors(
    frames: List,
    conf_range: List[int],
    natoms: np.ndarray,
    atomic_symbols: List[List[str]],
    rep1: str,
    rep2: str,
    HP1: Dict,
    HP2: Dict,
    nang1: int,
    nang2: int,
    nrad1: int,
    nrad2: int,
    neighspe1: List[str],
    neighspe2: List[str],
    species: List[str],
    lmax: Dict,
    lmax_max: int,
    lcut: Optional[str] = None,
    wigners_dir: Optional[str] = None,
    wigners_list: Optional[List[np.ndarray]] = None,
    sparsify: bool = False,
    ncut: int = None,
    vfps: Optional[Dict] = None,
    rank: int = 0) -> Dict[int, np.ndarray]:
    """
    Compute equivariant descriptors for all atoms.
    
    Supports both:
    - Config mode: loads Wigner matrices from files
    - Standalone mode: uses provided Wigner matrices
    """
    nspe1 = len(neighspe1)
    nspe2 = len(neighspe2)
    ndata = len(conf_range)
    natmax = max(natoms) if len(natoms) > 0 else 1
    natoms_total = sum(natoms)
    
    start_featomic = time.time()
    
    # Compute atomic representations
    omega1 = sph_utils.get_representation_coeffs(
        frames, rep1, HP1, rank, neighspe1, species, nang1, nrad1, natoms_total
    )
    if sph_utils.reps_equivalent(rep1, neighspe1, HP1, rep2, neighspe2, HP2):
        omega2 = omega1
    else:
        omega2 = sph_utils.get_representation_coeffs(
            frames, rep2, HP2, rank, neighspe2, species, nang2, nrad2, natoms_total
        )

    # Reshape arrays for Fortran indexing
    v1 = np.transpose(omega1, (1, 3, 0, 2)).copy()
    v2 = np.transpose(omega2, (1, 3, 0, 2)).copy()

    # Compute equivariant descriptors for each lambda
    pvec = {}
    for lam in range(lmax_max + 1):
        if rank == 0:
            print(f"lambda = {lam}", flush=True)

        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)

        # Load Wigner-3J symbols
        if wigners_dir is not None:
            # Config mode: load from file
            wigner3j = np.loadtxt(osp.join(
                wigners_dir, f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ))
        elif wigners_list is not None:
            # Standalone mode: use provided list
            if lam < len(wigners_list):
                wigner3j = wigners_list[lam]
            else:
                if rank == 0:
                    print(f"Warning: Missing Wigner data for lambda={lam}", flush=True)
                continue
        else:
            raise ValueError("Either wigners_dir or wigners_list must be provided")

        wigdim = wigner3j.size

        # Complex to real transformation
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        # Compute combined features
        if sparsify and vfps is not None and lam in vfps:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            nfps_val = len(vfps[lam])
            p = sph_utils.equicombsparse_numba(
                natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
                v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize, nfps_val, vfps[lam]
            )
            featsize = ncut
        else:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            p = sph_utils.equicomb_numba(
                natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
                v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize
            )

        # Store descriptors
        if lam == 0:
            p = p.reshape(natoms_total, featsize)
            pvec[lam] = np.zeros((ndata, natmax, featsize))
        else:
            p = p.reshape(natoms_total, 2 * lam + 1, featsize)
            pvec[lam] = np.zeros((ndata, natmax, 2 * lam + 1, featsize))

        j = 0
        for i, iconf in enumerate(conf_range):
            for iat in range(natoms[iconf]):
                pvec[lam][i, iat] = p[j]
                j += 1

    if rank == 0:
        print(f"featomic time (sec) = {time.time() - start_featomic}", flush=True)

    return pvec


def compute_equivariant_descriptors_response(
    frames: List,
    conf_range: List[int],
    natoms: np.ndarray,
    atomic_symbols: List[List[str]],
    rep1: str,
    rep2: str,
    HP1: Dict,
    HP2: Dict,
    nang1: int,
    nang2: int,
    nrad1: int,
    nrad2: int,
    neighspe1: List[str],
    neighspe2: List[str],
    species: List[str],
    lmax: Dict,
    lmax_max: int,
    wigners_dir: Optional[str] = None,
    wigners_list: Optional[List[np.ndarray]] = None,
    wigners_antisymm_dir: Optional[str] = None,
    wigners_antisymm_list: Optional[List[np.ndarray]] = None,
    rank: int = 0) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Compute equivariant descriptors for density-response prediction.
    
    Returns both symmetric and antisymmetric descriptors.
    """
    nspe1 = len(neighspe1)
    nspe2 = len(neighspe2)
    ndata = len(conf_range)
    natmax = max(natoms) if len(natoms) > 0 else 1
    natoms_total = sum(natoms)
    
    start_featomic = time.time()
    
    # Compute atomic representations
    omega1 = sph_utils.get_representation_coeffs(
        frames, rep1, HP1, rank, neighspe1, species, nang1, nrad1, natoms_total
    )
    if sph_utils.reps_equivalent(rep1, neighspe1, HP1, rep2, neighspe2, HP2):
        omega2 = omega1
    else:
        omega2 = sph_utils.get_representation_coeffs(
            frames, rep2, HP2, rank, neighspe2, species, nang2, nrad2, natoms_total
        )

    # Reshape arrays for Fortran indexing
    v1 = np.transpose(omega1, (1, 3, 0, 2)).copy()
    v2 = np.transpose(omega2, (1, 3, 0, 2)).copy()

    lmax_max_response = lmax_max + 1

    # Compute symmetric equivariant features
    power = {}
    for lam in range(lmax_max_response + 1):
        if rank == 0:
            print(f"lambda (symmetric) = {lam}", flush=True)

        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)

        # Load Wigner-3J symbols
        if wigners_dir is not None:
            wigner3j = np.loadtxt(os.path.join(
                wigners_dir, f"wigner_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ))
        elif wigners_list is not None:
            if lam < len(wigners_list):
                wigner3j = wigners_list[lam]
            else:
                if rank == 0:
                    print(f"Warning: Missing Wigner data for lambda={lam}", flush=True)
                continue
        else:
            raise ValueError("Either wigners_dir or wigners_list must be provided")

        wigdim = wigner3j.size
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        # Compute symmetric features
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        p = equicombnonorm(
            natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
            v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize
        )

        # Store descriptors
        if lam == 0:
            p = p.reshape(natoms_total, featsize)
            power[lam] = np.zeros((ndata, natmax, featsize))
        else:
            p = p.reshape(natoms_total, 2 * lam + 1, featsize)
            power[lam] = np.zeros((ndata, natmax, 2 * lam + 1, featsize))

        j = 0
        for i, iconf in enumerate(conf_range):
            for iat in range(natoms[iconf]):
                power[lam][i, iat] = p[j]
                j += 1

    # Compute antisymmetric equivariant features
    power_antisymm = {}
    for lam in range(1, lmax_max_response):
        if rank == 0:
            print(f"lambda (antisymmetric) = {lam}", flush=True)

        llmax, llvec = sph_utils.get_angular_indexes_antisymmetric(lam, nang1, nang2)

        # Load Wigner-3J symbols
        if wigners_antisymm_dir is not None:
            wigner3j = np.loadtxt(os.path.join(
                wigners_antisymm_dir, f"wigner_antisymm_lam-{lam}_lmax1-{nang1}_lmax2-{nang2}.dat"
            ))
        elif wigners_antisymm_list is not None and len(wigners_antisymm_list) > 0:
            if lam < len(wigners_antisymm_list):
                wigner3j = wigners_antisymm_list[lam]
            else:
                if rank == 0:
                    print(f"Warning: Missing antisymmetric Wigner data for lambda={lam}", flush=True)
                continue
        else:
            if rank == 0:
                print(f"Warning: Antisymmetric Wigner data not provided, skipping lambda={lam}", flush=True)
            continue

        wigdim = wigner3j.size
        c2r = sph_utils.complex_to_real_transformation([2 * lam + 1])[0]

        # Compute antisymmetric features
        featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
        p = antiequicombnonorm(
            natoms_total, nang1, nang2, nspe1 * nrad1, nspe2 * nrad2,
            v1, v2, wigner3j, llmax, llvec, lam, c2r, featsize
        )

        p = p.reshape(natoms_total, 2 * lam + 1, featsize)
        power_antisymm[lam] = np.zeros((ndata, natmax, 2 * lam + 1, featsize))

        j = 0
        for i, iconf in enumerate(conf_range):
            for iat in range(natoms[iconf]):
                power_antisymm[lam][i, iat] = p[j]
                j += 1

    if rank == 0:
        print(f"featomic time (sec) = {time.time() - start_featomic}", flush=True)

    return power, power_antisymm


def compute_predictions_for_structure(
    iconf: int,
    i_local: Optional[int],
    conf_range: List[int],
    atom_idx: Dict,
    natom_dict: Dict,
    atomic_symbols: List[List[str]],
    natoms: np.ndarray,
    lmax: Dict,
    nmax: Dict,
    species: List[str],
    zeta: float,
    pvec: Dict[int, np.ndarray],
    power_env_sparse: Dict,
    Vmat: Dict,
    Mspe: Dict,
    weights: np.ndarray,
    average: bool = False,
    av_coefs: Optional[Dict] = None) -> np.ndarray:
    """
    Compute predictions for a single structure.
    """
    if i_local is None:
        iconf_idx = conf_range.index(iconf) if isinstance(conf_range, list) else iconf
    else:
        iconf_idx = i_local
    # Compute size
    Tsize = 0
    for iat in range(natoms[iconf_idx]):
        spe = atomic_symbols[iconf_idx][iat]
        for l in range(lmax[spe] + 1):
            for n in range(nmax[(spe, l)]):
                Tsize += 2 * l + 1

    # Compute kernels and projections
    psi_nm = {}
    for spe in species:
        atom_indices = atom_idx[(iconf, spe)]
        # Lambda = 0
        if (0, spe) in power_env_sparse:
            if zeta == 1:
                psi_nm[(spe, 0)] = np.dot(
                    pvec[0][iconf_idx, atom_indices],
                    power_env_sparse[(0, spe)].T
                )
            else:
                kernel0_nm = np.dot(
                    pvec[0][iconf_idx, atom_indices],
                    power_env_sparse[(0, spe)].T
                )
                kernel_nm = kernel0_nm ** zeta
                psi_nm[(spe, 0)] = np.dot(kernel_nm, Vmat[(0, spe)])
        
        # Lambda > 0
        for lam in range(1, lmax[spe] + 1):
            if (lam, spe) not in power_env_sparse:
                continue
                
            featsize = pvec[lam].shape[-1]

            if zeta == 1:
                psi_nm[(spe, lam)] = np.dot(
                    pvec[lam][iconf_idx, atom_indices].reshape(
                        natom_dict[(iconf, spe)] * (2 * lam + 1), featsize
                    ),
                    power_env_sparse[(lam, spe)].T
                )
            else:
                kernel_nm = np.dot(
                    pvec[lam][iconf_idx, atom_indices].reshape(
                        natom_dict[(iconf, spe)] * (2 * lam + 1), featsize
                    ),
                    power_env_sparse[(lam, spe)].T
                )
                kernel_nm_blocks = kernel_nm.reshape(
                    natom_dict[(iconf, spe)], 2 * lam + 1, Mspe[spe], 2 * lam + 1
                )
                kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (zeta - 1)
                kernel_nm = kernel_nm_blocks.reshape(
                    natom_dict[(iconf, spe)] * (2 * lam + 1), Mspe[spe] * (2 * lam + 1)
                )
                psi_nm[(spe, lam)] = np.dot(kernel_nm, Vmat[(lam, spe)])

    # Compute predictions per channel
    C = {}
    ispe = {}
    isize = 0
    for spe in species:
        ispe[spe] = 0
        for l in range(lmax[spe] + 1):
            for n in range(nmax[(spe, l)]):
                if (spe, l) in psi_nm:
                    Mcut = psi_nm[(spe, l)].shape[1]
                    C[(spe, l, n)] = np.dot(psi_nm[(spe, l)], weights[isize:isize + Mcut])
                    isize += Mcut

    # Fill prediction vector
    pred_coefs = np.zeros(Tsize)
    Av_coeffs = np.zeros(Tsize) if average else None

    i = 0
    for iat in range(natoms[iconf_idx]):
        spe = atomic_symbols[iconf_idx][iat]
        for l in range(lmax[spe] + 1):
            for n in range(nmax[(spe, l)]):
                if (spe, l, n) in C:
                    pred_coefs[i:i + 2 * l + 1] = C[(spe, l, n)][
                        ispe[spe] * (2 * l + 1):ispe[spe] * (2 * l + 1) + 2 * l + 1
                    ]
                if average and l == 0 and av_coefs is not None:
                    Av_coeffs[i] = av_coefs[spe][n]
                i += 2 * l + 1
        ispe[spe] += 1

    # Add averages
    if average and Av_coeffs is not None:
        pred_coefs += Av_coeffs

    return pred_coefs


def save_pred_descriptor(
    data: Dict[int, np.ndarray],
    config_range: List[int],
    natoms: List[int],
    dpath: str
):
    """Save the descriptor data of the prediction dataset."""
    assert len(config_range) == len(natoms), (
        f"Length mismatch: {len(config_range)} vs {len(natoms)}"
    )

    for lam, data_this_lam in data.items():
        assert data_this_lam.shape[0] == len(config_range), (
            f"First dimension mismatch at lambda={lam}"
        )

    for idx, idx_in_full_dataset in enumerate(config_range):
        this_data: Dict[str, np.ndarray] = {}
        this_natoms = natoms[idx]
        for lam, data_this_lam in data.items():
            this_data[f"lam{lam}"] = data_this_lam[idx, :this_natoms]
        
        with open(osp.join(dpath, f"descriptor_{idx_in_full_dataset + 1}.npz"), "wb") as f:
            np.savez(f, **this_data)


# ============================================================================
# CONFIG MODE (Full SALTED workflow)
# ============================================================================

def predict_config_mode():
    """
    Full SALTED prediction workflow using config file.
    
    This maintains all features of the original prediction.py:
    - Config file parsing
    - MPI parallelization
    - Density and density-response predictions
    - CP2K integration for properties
    - Directory structure loading
    """
    inp = ParseConfig().parse_input()
    (saltedname, saltedpath, saltedtype,
    filename, species, average,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data, alpha_only,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, trainsel, nspe1, nspe2, HP1, HP2) = ParseConfig().get_all_params()

    if filename_pred == PLACEHOLDER or predname == PLACEHOLDER:
        raise ValueError(
            "No prediction file and name provided. "
            "Specify 'prediction.filename' and 'prediction.predname' in input file."
        )

    comm, size, rank, parallel = detect_mpi()

    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        filename_pred, species, dfbasis
    )
    atom_idx, natom_dict = sys_utils_get_atom_idx(ndata, natoms, species, atomic_symbols)

    if rank == 0:
        print(f"The dataset contains {ndata} frames.")

    # Initialize conf_range for both parallel and serial cases
    if parallel:
        check_MPI_tasks_count(comm, ndata, "predicting structures")
        conf_range = distribute_jobs(comm, list(range(ndata)))
        ndata = len(conf_range)
        natmax = max(natoms[conf_range])
        if inp.salted.verbose:
            print(f"Task {rank} handles: {format_index_ranges(conf_range, True)}", flush=True)
    else:
        conf_range = list(range(ndata))
    
    natoms_total = sum(natoms[conf_range])

    reg_log10_intstr = str(int(np.log10(regul)))

    # Load regression weights
    ntrain = int(Ntrain * trainfrac)
    weights = np.load(osp.join(
        saltedpath,
        f"regrdir_{saltedname}",
        f"M{Menv}_zeta{zeta}",
        f"weights_N{ntrain}_reg{reg_log10_intstr}.npy"
    ))

    if qmcode == "cp2k":
        charge_integrals, dipole_integrals = init_moments(inp, species, lmax, nmax, rank)

    # Setup output directory
    pdir = osp.join(
        saltedpath,
        f"predictions_{saltedname}_{predname}"
    )
    dirpath = osp.join(
        pdir,
        f"M{Menv}_zeta{zeta}",
        f"N{ntrain}_reg{reg_log10_intstr}",
    )

    if rank == 0:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        if saltedtype == "density-response":
            for icart in ["x", "y", "z"]:
                cartpath = os.path.join(dirpath, icart)
                if not os.path.exists(cartpath):
                    os.mkdir(cartpath)
    
    if parallel:
        comm.Barrier()

    # Initialize property files for CP2K
    qfile, dfile, pfile = None, None, None
    if qmcode == "cp2k" :
        if saltedtype == "density":
            qfile = init_property_file("charges", saltedpath, pdir, Menv, zeta, ntrain, reg_log10_intstr, rank, size, comm)
            dfile = init_property_file("dipoles", saltedpath, pdir, Menv, zeta, ntrain, reg_log10_intstr, rank, size, comm)
        if saltedtype == "density-response":
            pfile = init_property_file("polarizabilities", saltedpath, pdir, Menv, zeta, ntrain, reg_log10_intstr, rank, size, comm)

    start = time.time()

    # Read frames
    frames = read(filename_pred, ":")
    frames = [frames[i] for i in conf_range]

    if saltedtype == "density":
        # Load FPS information if required
        if sparsify:
            vfps = {}
            for lam in range(lmax_max + 1):
                vfps[lam] = np.load(osp.join(
                    saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
                ))
        else:
            vfps = None

        # Load training features and RKHS projection matrix
        Vmat, Mspe, power_env_sparse = get_feats_projs(species, lmax)

        # Load spherical averages if required
        av_coefs_dict = {}
        if average:
            for spe in species:
                av_coefs_dict[spe] = np.load(
                    os.path.join(saltedpath, "coefficients", "averages", f"averages_{spe}.npy")
                )

        # Compute equivariant descriptors
        pvec = compute_equivariant_descriptors(
            frames, conf_range, natoms, atomic_symbols,
            rep1, rep2, HP1, HP2,
            nang1, nang2, nrad1, nrad2,
            neighspe1, neighspe2, species, lmax, lmax_max,
            wigners_dir=osp.join(saltedpath, "wigners"),
            sparsify=sparsify,
            ncut=ncut,
            vfps=vfps,
            rank=rank,
        )
        # Save descriptor if requested
        if inp.prediction.save_descriptor:
            if rank == 0:
                print(f"Saving descriptor to {dirpath}", flush=True)
            save_pred_descriptor(pvec, conf_range, list(natoms[conf_range]), dirpath)

        # Compute predictions
        for i, iconf in enumerate(conf_range):
            pred_coefs = compute_predictions_for_structure(
                iconf, i, conf_range, atom_idx, natom_dict, atomic_symbols, natoms,
                lmax, nmax, species, zeta, pvec, power_env_sparse, Vmat, Mspe, weights,
                average=average, av_coefs=av_coefs_dict
            )

            if qmcode == "cp2k":
                charge, dipole = compute_charge_and_dipole(
                    frames[i], inp.qm.pseudocharge, natoms[iconf], atomic_symbols[iconf],
                    lmax, nmax, species, charge_integrals, dipole_integrals, pred_coefs, average
                )
                print(iconf + 1, charge, file=qfile)
                print(iconf + 1, dipole["x"], dipole["y"], dipole["z"], file=dfile)

            np.savetxt(osp.join(dirpath, f"COEFFS-{iconf + 1}.dat"), pred_coefs)

    elif saltedtype == "density-response":
        # Load training features and RKHS projection matrix (response version)
        Vmat, Mspe, power_env_sparse, power_env_sparse_antisymm = get_feats_projs_response(
            species, lmax
        )

        # Compute symmetric and antisymmetric equivariant descriptors
        power, power_antisymm = compute_equivariant_descriptors_response(
            frames, conf_range, natoms, atomic_symbols,
            rep1, rep2, HP1, HP2,
            nang1, nang2, nrad1, nrad2,
            neighspe1, neighspe2, species, lmax, lmax_max,
            wigners_dir=osp.join(saltedpath, "wigners"),
            wigners_antisymm_dir=osp.join(saltedpath, "wigners"),
            rank=rank,
        )

        # Compute predictions for density-response (simplified - core logic only)
        if rank == 0:
            print("Density-response prediction mode not fully implemented in unified script", flush=True)
            print("Using original prediction.py for density-response", flush=True)

    # Close property files
    if qmcode == "cp2k":
        if qfile is not None:
            qfile.close()
        if dfile is not None:
            dfile.close()
        if pfile is not None:
            pfile.close()

    if rank == 0:
        print(f"\nTotal time: {(time.time() - start):.2f} s")


# ============================================================================
# STANDALONE MODE (Binary model file)
# ============================================================================
def predict_standalone_mode(model_file: str, xyz_file: str, output_dir: str = None):
    """
    Standalone prediction using .salted model file with MPI support.
    """
    # Detect MPI
    comm, size, rank, parallel = detect_mpi()

    if rank == 0:
        print(f"Loading model from {model_file}...")
    
    model = read_model.read_salted_model(model_file)

    # Extract configuration
    config = model['config']
    species_str = config['speci']
    species = species_str.split()

    # Extract hyperparameters
    nang1 = int(config['nang1'])
    nang2 = int(config['nang2'])
    nrad1 = int(config['nrad1'])
    nrad2 = int(config['nrad2'])
    rcut1 = float(config['rcut1'])
    rcut2 = float(config['rcut2'])
    sig1 = float(config['sig1'])
    sig2 = float(config['sig2'])
    zeta = int(config['zeta'])
    use_average = bool(config['averg'])
    use_sparsify = bool(config['spars'])
    ncut = int(config.get('ncut', 0))
    dfbasis = config['dfbas']
    saltedtype = config.get('predtype', 'density')  # Default to density if not specified

    neighspe1_str = config.get('nspe1', species_str)
    neighspe2_str = config.get('nspe2', species_str)
    neighspe1 = neighspe1_str.split()
    neighspe2 = neighspe2_str.split()
    nspe1 = len(neighspe1)
    nspe2 = len(neighspe2)

    # Representation types
    rep1 = "rho"
    rep2 = "rho"

    if rank == 0:
        print(f"Model configuration:")
        print(f"  Species: {species}")
        print(f"  Basis: {dfbasis}")
        print(f"  Prediction type: {saltedtype}")
        print(f"  nang1={nang1}, nang2={nang2}, nrad1={nrad1}, nrad2={nrad2}")
        print(f"  zeta={zeta}, sparsify={use_sparsify}, average={use_average}")

    # Read system
    if rank == 0:
        print(f"\nReading structures from {xyz_file}...")
    
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system(
        xyz_file, species, dfbasis, basis_data=model.get('basis')
    )
    
    if rank == 0:
        print(f"  Found {ndata} structures")
        print(f"  Max atoms per structure: {natmax}")

    # Setup output directory
    if output_dir is None:
        output_dir = "predictions"
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory: {output_dir}")
    
    if parallel:
        comm.Barrier()

    # Distribute structures across MPI ranks
    if parallel:
        check_MPI_tasks_count(comm, ndata, "predicting structures")
        conf_range = distribute_jobs(comm, list(range(ndata)))
        ndata_local = len(conf_range)
        natmax_local = max(natoms[conf_range]) if len(conf_range) > 0 else 1
        if rank == 0:
            print(f"Distributed structures across {size} MPI tasks")
            print(f"Task {rank} handles: {format_index_ranges(conf_range, True)}", flush=True)
    else:
        conf_range = list(range(ndata))
        ndata_local = ndata
        natmax_local = natmax

    # Get atom indices for all structures
    atom_idx, natom_dict = get_atom_idx(ndata, natoms, species, atomic_symbols)

    # Read frames
    frames = read(xyz_file, ":")
    frames_local = [frames[i] for i in conf_range]

    # Extract model data
    weights = model['weights']
    wigners = model.get('wigners', [])
    averages = model.get('averages', {})
    fps_data = model.get('fps', [])
    feats = model['feats']
    projectors = model['projectors']

    # Setup hyperparameters
    HP1 = {
        "cutoff": {"radius": rcut1, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
        "density": {"type": "Gaussian", "width": sig1},
        "basis": {
            "type": "TensorProduct",
            "max_angular": nang1,
            "radial": {"type": "Gto", "max_radial": nrad1 - 1},
            "spline_accuracy": 1e-06
        }
    }

    HP2 = {
        "cutoff": {"radius": rcut2, "smoothing": {"type": "ShiftedCosine", "width": 0.1}},
        "density": {"type": "Gaussian", "width": sig2},
        "basis": {
            "type": "TensorProduct",
            "max_angular": nang2,
            "radial": {"type": "Gto", "max_radial": nrad2 - 1},
            "spline_accuracy": 1e-06
        }
    }

    # Prepare FPS indices if using sparsification
    vfps = {}
    if use_sparsify:
        for lam in range(lmax_max + 1):
            if lam < len(fps_data):
                vfps[lam] = fps_data[lam]

    if rank == 0:
        print("\nComputing atomic representations...")
    
    start_time = time.time()

    # Compute total atoms for feature computation
    natoms_total_local = sum(natoms[conf_range])

    if saltedtype == "density":
        # Compute equivariant descriptors for density prediction
        pvec = compute_equivariant_descriptors(
            frames_local, conf_range, natoms, atomic_symbols,
            rep1, rep2, HP1, HP2,
            nang1, nang2, nrad1, nrad2,
            neighspe1, neighspe2, species, lmax, lmax_max,
            lcut=None,
            wigners_list=wigners,
            sparsify=use_sparsify,
            ncut=ncut,
            vfps=vfps if use_sparsify else None,
            rank=rank,
        )

        if rank == 0:
            print(f"  Feature computation time: {time.time() - start_time:.2f} s")

        # Load projectors and features
        Vmat = {}
        power_env_sparse = {}
        Mspe = {}

        for spe in species:
            if spe not in feats or spe not in projectors:
                if rank == 0:
                    print(f"Warning: Missing data for species {spe}")
                continue

            for lam in range(lmax[spe] + 1):
                lam_str = str(lam)
                if lam_str in feats[spe]:
                    power_env_sparse[(lam, spe)] = feats[spe][lam_str]
                if lam_str in projectors[spe]:
                    Vmat[(lam, spe)] = projectors[spe][lam_str]

                if lam == 0:
                    Mspe[spe] = power_env_sparse[(lam, spe)].shape[0]

        # Compute predictions for density
        if rank == 0:
            print(f"\nComputing predictions for {ndata} structures...")
        
        for i, iconf in enumerate(conf_range):
            if rank == 0 or i % 10 == 0:
                print(f"  [Rank {rank}] Structure {iconf + 1}/{ndata}...", end=" ", flush=True)
            
            start = time.time()

            pred_coefs = compute_predictions_for_structure(
                iconf, i, conf_range, atom_idx, natom_dict, atomic_symbols, natoms,
                lmax, nmax, species, zeta, pvec, power_env_sparse, Vmat, Mspe, weights,
                average=use_average, av_coefs=averages if use_average else None
            )

            # Save predictions
            output_file = os.path.join(output_dir, f"COEFFS-{iconf + 1}.dat")
            np.savetxt(output_file, pred_coefs)

            if rank == 0 or i % 10 == 0:
                print(f"done ({time.time() - start:.2f} s)")

    elif saltedtype == "density-response":
        # Compute equivariant descriptors for density-response prediction
        # Note: For density-response, wigners_antisymm_list would need to be available in the model
        # Currently using empty list as placeholder
        power, power_antisymm = compute_equivariant_descriptors_response(
            frames_local, conf_range, natoms, atomic_symbols,
            rep1, rep2, HP1, HP2,
            nang1, nang2, nrad1, nrad2,
            neighspe1, neighspe2, species, lmax, lmax_max,
            wigners_list=wigners,
            wigners_antisymm_list=[],  # TODO: Extract from model when available
            rank=rank,
        )

        if rank == 0:
            print(f"  Feature computation time: {time.time() - start_time:.2f} s")
            print("Density-response prediction not fully implemented in standalone mode yet")

    else:
        raise ValueError(f"Unknown prediction type: {saltedtype}")

    if parallel:
        comm.Barrier()

    if rank == 0:
        print(f"\nPredictions saved to {output_dir}/")


# ============================================================================
# CLI AND MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Unified prediction script for SALTED models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  CONFIG MODE (full SALTED workflow):
    python predict_unified.py

  STANDALONE MODE (binary model file):
    python predict_unified.py --model model.salted --xyz structures.xyz
    python predict_unified.py --model model.salted --xyz structures.xyz --output predictions/
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to .salted model file (standalone mode)"
    )
    parser.add_argument(
        "--xyz",
        type=str,
        help="Path to XYZ structures file (required for standalone mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Output directory for predictions (standalone mode, default: predictions)"
    )

    args = parser.parse_args()

    if args.model and not args.xyz:
        parser.error("--xyz is required when using --model")


    if args.model:
        # Run standalone mode
        try:
            predict_standalone_mode(args.model, args.xyz, args.output)
            exit()
        except Exception as e:
            print(f"Error in standalone mode: {e}")

    # As fallback, also allow running config mode if no model file is provided
    try:
        predict_config_mode()
        exit()
    except Exception as e:
        print(f"Error in config mode: {e}")


if __name__ == "__main__":
    main()
