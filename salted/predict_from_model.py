#!/usr/bin/env python
"""
Standalone prediction script that uses only a .salted model file and an xyz file
to predict density expansion coefficients.

Usage:
    python predict_from_model.py <model.salted> <structures.xyz> [output_dir]
"""

import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, Tuple, List
from ase.io import read
from ase.data import atomic_numbers

import read_model

# Import SALTED modules for feature computation
try:
    from salted import sph_utils
    from salted.lib import equicomb, equicombsparse
    from salted import basis
except ImportError:
    print("Error: SALTED package not found. Please install SALTED.")
    sys.exit(1)


def read_system_from_xyz(filename: str, species: List[str], dfbasis: str):
    """
    Read geometry file and return formatted system information.
    
    Parameters
    ----------
    filename : str
        Path to XYZ file
    species : list of str
        List of atomic species to include
    dfbasis : str
        Density fitting basis set name
        
    Returns
    -------
    tuple
        (species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax)
    """
    # Read basis
    lmax, nmax = basis.basiset(dfbasis)
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe] + 1):
            nlist.append(nmax[(spe, l)])
    nnmax = max(nlist)
    lmax_max = max(llist)

    # Read structures
    frames = read(filename, ":", parallel=False)
    ndata = len(frames)

    # Extract atomic symbols, filtering by species list
    atomic_symbols = []
    natoms = np.zeros(ndata, int)
    for iconf in range(ndata):
        symbols = frames[iconf].get_chemical_symbols()
        # Filter out species not in our list
        filtered_symbols = [s for s in symbols if s in species]
        atomic_symbols.append(filtered_symbols)
        natoms[iconf] = len(filtered_symbols)

    natmax = max(natoms)

    return species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax


def get_atom_idx(ndata: int, natoms: np.ndarray, species: List[str], atomic_symbols: List[List[str]]):
    """Get indices of atoms by species for each configuration."""
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


def compute_predictions(model_file: str, xyz_file: str, output_dir: str = None):
    """
    Compute predictions using a SALTED model and XYZ structures.
    
    Parameters
    ----------
    model_file : str
        Path to .salted model file
    xyz_file : str
        Path to XYZ file with structures
    output_dir : str, optional
        Directory for output files (default: current directory)
    """
    
    print(f"Loading model from {model_file}...")
    model = read_model.read_salted_model(model_file)
    
    # Extract configuration from model
    config = model['config']
    species_str = config['speci']
    species = species_str.split()
    
    # Extract hyperparameters
    nang1 = config['nang1']
    nang2 = config['nang2']
    nrad1 = config['nrad1']
    nrad2 = config['nrad2']
    rcut1 = config['rcut1']
    rcut2 = config['rcut2']
    sig1 = config['sig1']
    sig2 = config['sig2']
    zeta = config['zeta']
    use_average = config['averg']
    use_sparsify = config['spars']
    ncut = config.get('ncut', 0)
    dfbasis = config['dfbas']
    
    neighspe1_str = config.get('nspe1', species_str)
    neighspe2_str = config.get('nspe2', species_str)
    neighspe1 = neighspe1_str.split()
    neighspe2 = neighspe2_str.split()
    nspe1 = len(neighspe1)
    nspe2 = len(neighspe2)
    
    # Representation types (density expansion uses "rho")
    rep1 = "rho"
    rep2 = "rho"
    
    print(f"Model configuration:")
    print(f"  Species: {species}")
    print(f"  Basis: {dfbasis}")
    print(f"  nang1={nang1}, nang2={nang2}, nrad1={nrad1}, nrad2={nrad2}")
    print(f"  rcut1={rcut1}, rcut2={rcut2}, sig1={sig1}, sig2={sig2}")
    print(f"  zeta={zeta}, sparsify={use_sparsify}, average={use_average}")
    
    # Read system
    print(f"\nReading structures from {xyz_file}...")
    species, lmax, nmax, lmax_max, nnmax, ndata, atomic_symbols, natoms, natmax = read_system_from_xyz(
        xyz_file, species, dfbasis
    )
    print(f"  Found {ndata} structures")
    print(f"  Max atoms per structure: {natmax}")
    
    # Get atom indices
    atom_idx, natom_dict = get_atom_idx(ndata, natoms, species, atomic_symbols)
    
    # Read frames
    frames = read(xyz_file, ":")
    
    # Setup output directory
    if output_dir is None:
        output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory: {output_dir}")
    
    # Extract model data
    weights = model['weights']
    wigners = model['wigners']
    averages = model.get('averages', {})
    fps_data = model.get('fps', [])
    feats = model['feats']
    projectors = model['projectors']
    
    # Compute features
    print("\nComputing atomic representations...")
    start_time = time.time()
    
    # Setup hyperparameters for density representation
    HYPER_PARAMETERS_DENSITY = {
        "cutoff": {
            "radius": rcut1,
            "smoothing": {
                "type": "ShiftedCosine",
                "width": 0.1
            }
        },
        "density": {
            "type": "Gaussian",
            "width": sig1
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": nang1,
            "radial": {
                "type": "Gto",
                "max_radial": nrad1 - 1
            },
            "spline_accuracy": 1e-06
        }
    }
    
    HYPER_PARAMETERS_POTENTIAL = {
        "density": {
            "type": "SmearedPowerLaw",
            "smearing": sig2,
            "exponent": 1
        },
        "basis": {
            "type": "TensorProduct",
            "max_angular": nang2,
            "radial": {
                "type": "Gto",
                "max_radial": nrad2 - 1,
                "radius": rcut2
            },
            "spline_accuracy": 1e-06
        }
    }
    
    natoms_total = sum(natoms)
    
    # Compute SOAP-like representations
    omega1 = sph_utils.get_representation_coeffs(
        frames, rep1, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL,
        0, neighspe1, species, nang1, nrad1, natoms_total
    )
    omega2 = sph_utils.get_representation_coeffs(
        frames, rep2, HYPER_PARAMETERS_DENSITY, HYPER_PARAMETERS_POTENTIAL,
        0, neighspe2, species, nang2, nrad2, natoms_total
    )
    
    # Reshape for Fortran indexing
    v1 = np.transpose(omega1, (2, 0, 3, 1))
    v2 = np.transpose(omega2, (2, 0, 3, 1))
    
    print(f"  Feature computation time: {time.time() - start_time:.2f} s")
    
    # Prepare FPS indices if using sparsification
    vfps = {}
    if use_sparsify:
        for lam in range(lmax_max + 1):
            if lam < len(fps_data):
                vfps[lam] = fps_data[lam]
            else:
                print(f"Warning: FPS data missing for lambda={lam}")
    
    # Load projectors and features into dictionaries
    Vmat = {}
    power_env_sparse = {}
    Mspe = {}
    
    for spe in species:
        if spe not in feats or spe not in projectors:
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
            
            # Precompute projection if linear (zeta=1)
            if zeta == 1 and (lam, spe) in power_env_sparse and (lam, spe) in Vmat:
                power_env_sparse[(lam, spe)] = np.dot(
                    Vmat[(lam, spe)].T, power_env_sparse[(lam, spe)]
                )
    
    # Compute predictions for each structure
    print(f"\nComputing predictions for {ndata} structures...")
    
    # First compute all equivariant descriptors for all atoms at once
    print("  Computing equivariant descriptors...")
    pvec = {}
    
    for lam in range(lmax_max + 1):
        # Get angular indices
        llmax, llvec = sph_utils.get_angular_indexes_symmetric(lam, nang1, nang2)
        
        # Get Wigner symbols
        if lam < len(wigners):
            wigner3j = wigners[lam]
        else:
            print(f"Warning: Missing Wigner data for lambda={lam}")
            continue
        
        wigdim = wigner3j.size
        
        # Complex to real transformation
        c2r = sph_utils.complex_to_real_transformation([2*lam + 1])[0]
        
        # Compute combined features for ALL atoms
        if use_sparsify and lam in vfps:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            nfps = len(vfps[lam])
            p = equicombsparse.equicombsparse(
                natoms_total, nang1, nang2, nspe1*nrad1, nspe2*nrad2,
                v1, v2, wigdim, wigner3j, llmax, llvec.T, lam, c2r,
                featsize, nfps, vfps[lam]
            )
            p = np.transpose(p, (2, 0, 1))
            featsize = ncut
        else:
            featsize = nspe1 * nspe2 * nrad1 * nrad2 * llmax
            p = equicomb.equicomb(
                natoms_total, nang1, nang2, nspe1*nrad1, nspe2*nrad2,
                v1, v2, wigdim, wigner3j, llmax, llvec.T, lam, c2r, featsize
            )
            p = np.transpose(p, (2, 0, 1))
        
        # Reshape and store
        if lam == 0:
            p = p.reshape(natoms_total, featsize)
            pvec[lam] = np.zeros((ndata, natmax, featsize))
        else:
            p = p.reshape(natoms_total, 2*lam + 1, featsize)
            pvec[lam] = np.zeros((ndata, natmax, 2*lam + 1, featsize))
        
        # Distribute to structures
        j = 0
        for iconf in range(ndata):
            for iat in range(natoms[iconf]):
                pvec[lam][iconf, iat] = p[j]
                j += 1
    
    print("  Equivariant descriptors computed.")
    
    # Now compute predictions for each structure
    for iconf in range(ndata):
        print(f"  Structure {iconf + 1}/{ndata}...", end=" ", flush=True)
        start = time.time()
        
        # Compute kernel and predictions
        psi_nm = {}
        
        for spe in species:
            atom_indices = atom_idx[(iconf, spe)]
            # Lambda = 0
            if (0, spe) in power_env_sparse:
                if zeta == 1:
                    psi_nm[(spe, 0)] = np.dot(
                        pvec[0][iconf, atom_indices], power_env_sparse[(0, spe)].T
                    )
                else:
                    kernel0_nm = np.dot(
                        pvec[0][iconf, atom_indices], power_env_sparse[(0, spe)].T
                    )
                    kernel_nm = kernel0_nm ** zeta
                    psi_nm[(spe, 0)] = np.dot(kernel_nm, Vmat[(0, spe)])
            
            # Lambda > 0
            for lam in range(1, lmax[spe] + 1):
                featsize = pvec[lam].shape[-1]
                
                if zeta == 1:
                    psi_nm[(spe, lam)] = np.dot(
                        pvec[lam][iconf, atom_indices].reshape(natom_dict[(iconf, spe)] * (2*lam + 1), featsize),
                        power_env_sparse[(lam, spe)].T
                    )
                else:
                    kernel_nm = np.dot(
                        pvec[lam][iconf, atom_indices].reshape(natom_dict[(iconf, spe)] * (2*lam + 1), featsize),
                        power_env_sparse[(lam, spe)].T
                    )
                    kernel_nm_blocks = kernel_nm.reshape(
                        natom_dict[(iconf, spe)], 2*lam + 1, Mspe[spe], 2*lam + 1
                    )
                    kernel_nm_blocks *= kernel0_nm[:, np.newaxis, :, np.newaxis] ** (zeta - 1)
                    kernel_nm = kernel_nm_blocks.reshape(
                        natom_dict[(iconf, spe)] * (2*lam + 1), Mspe[spe] * (2*lam + 1)
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
                    Mcut = psi_nm[(spe, l)].shape[1]
                    C[(spe, l, n)] = np.dot(psi_nm[(spe, l)], weights[isize:isize + Mcut])
                    isize += Mcut
 
        # Compute total size needed
        Tsize = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    Tsize += 2*l + 1
        
        # Fill prediction vector
        pred_coefs = np.zeros(Tsize)
        av_coefs = np.zeros(Tsize) if use_average else None
        
        i = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            for l in range(lmax[spe] + 1):
                for n in range(nmax[(spe, l)]):
                    if (spe, l, n) in C:
                        pred_coefs[i:i + 2*l + 1] = C[(spe, l, n)][ispe[spe] * (2*l + 1):ispe[spe] * (2*l + 1) + 2*l + 1]
                    if use_average and l == 0 and av_coefs is not None:
                        av_coefs[i] = averages[spe][n]
                    i += 2*l + 1
            ispe[spe] += 1
        
        # Add averages if required
        if use_average and av_coefs is not None:
            pred_coefs += av_coefs
        
        # Save predictions
        output_file = os.path.join(output_dir, f"COEFFS-{iconf + 1}.dat")
        # np.savetxt(output_file, pred_coefs)
        np.save(output_file.replace('.dat', '.npy'), pred_coefs)
        
        print(f"done ({time.time() - start:.2f} s)")
    
    print(f"\nPredictions saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Predict density expansion coefficients using a SALTED model"
    )
    parser.add_argument("model", help="Path to .salted model file")
    parser.add_argument("xyz", help="Path to XYZ structures file")
    parser.add_argument("-o", "--output", default="predictions", 
                       help="Output directory (default: predictions)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found")
        sys.exit(1)
    
    if not os.path.exists(args.xyz):
        print(f"Error: XYZ file '{args.xyz}' not found")
        sys.exit(1)
    
    compute_predictions(args.model, args.xyz, args.output)


if __name__ == "__main__":
    main()
