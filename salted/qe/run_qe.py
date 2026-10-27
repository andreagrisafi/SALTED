"""Run Quantum ESPRESSO (pw.x) SCF calculations to generate electron densities for SALTED.

Unlike CP2K -- an external binary the user runs themselves outside SALTED, see
`salted.cp2k.xyz2sys` -- this module drives `pw.x` directly through ASE's `Espresso` calculator.
`salted.qe.xyz2sys` must be run first to create the per-structure `conf_N/structure.xyz` folders
this module reads from. `salted.qe.qe2cube` then extracts the converged density as a Gaussian
cube file via `pp.x`.

Required `inp.qm` fields when `qmcode: qe` (see `salted.sys_utils.ParseConfig`):
    pseudo_dir: directory containing the UPF pseudopotential files
    pseudopotentials: mapping of species -> UPF filename, e.g. {"O": "O.upf", "H": "H.upf"}
    ecutwfc: plane-wave cutoff (Ry)
    kpts: Monkhorst-Pack grid as [nx, ny, nz], or the string "gamma"
Optional:
    ecutrho, occupations ("fixed"/"smearing"/...), smearing, degauss: passed through to pw.x's
        &SYSTEM namelist when given
    nproc: number of MPI ranks pw.x itself is launched with per structure (default: 1, i.e. no
        mpirun wrapper). This is independent of, and composes with, running this script itself
        under mpirun to distribute structures across ranks (see salted.sys_utils.detect_mpi).
    pwx_command: override the pw.x executable/path (default "pw.x")

Outputs land alongside `structure.xyz` in each `conf_N` folder: ASE's `espresso.pwi`/`espresso.pwo`,
and the QE `outdir` (`PWSCF_OUTDIR`, holding `PWSCF_PREFIX.save/` with the converged density) that
`salted.qe.qe2cube` reads from.
"""

import argparse
import os.path as osp

from ase.calculators.espresso import Espresso, EspressoProfile
from ase.io import read

from salted.qe.xyz2sys import STRUCTURE_FNAME
from salted.sys_utils import (
    ARGHELP_INDEX_STR,
    PLACEHOLDER,
    ParseConfig,
    check_MPI_tasks_count,
    detect_mpi,
    distribute_jobs,
    parse_index_str,
)

PWSCF_PREFIX = "pwscf"
PWSCF_OUTDIR = "out"


def default_if_placeholder(val, default):
    return default if val == PLACEHOLDER else val


def pw_command(inp) -> str:
    pwx = default_if_placeholder(inp.qm.pwx_command, "pw.x")
    nproc = default_if_placeholder(inp.qm.nproc, 1)
    return f"mpirun -np {nproc} {pwx}" if nproc > 1 else pwx


def build_input_data(inp) -> dict:
    """Assemble the pw.x &CONTROL/&SYSTEM/&ELECTRONS namelists from the SALTED input file."""
    control = {
        "calculation": "scf",
        "prefix": PWSCF_PREFIX,
        "outdir": PWSCF_OUTDIR,
    }

    system = {"ecutwfc": inp.qm.ecutwfc}
    ecutrho = default_if_placeholder(inp.qm.ecutrho, None)
    if ecutrho is not None:
        system["ecutrho"] = ecutrho

    occupations = default_if_placeholder(inp.qm.occupations, None)
    if occupations is not None:
        system["occupations"] = occupations
        if occupations == "smearing":
            system["smearing"] = default_if_placeholder(inp.qm.smearing, "gaussian")
            system["degauss"] = default_if_placeholder(inp.qm.degauss, 0.01)

    return {"control": control, "system": system, "electrons": {}}


def run_one(inp, conf_dir: str):
    """Run a single pw.x SCF calculation inside `conf_dir`, reading its `structure.xyz`."""
    atoms = read(osp.join(conf_dir, STRUCTURE_FNAME))

    species = sorted(set(atoms.get_chemical_symbols()))
    pseudo_map = inp.qm.pseudopotentials.to_dict()
    missing = [spe for spe in species if spe not in pseudo_map]
    if missing:
        raise ValueError(f"No pseudopotential configured for species {missing} (qm.pseudopotentials)")
    pseudopotentials = {spe: pseudo_map[spe] for spe in species}

    kpts = None if inp.qm.kpts == "gamma" else tuple(inp.qm.kpts)

    profile = EspressoProfile(command=pw_command(inp), pseudo_dir=inp.qm.pseudo_dir)
    atoms.calc = Espresso(
        profile=profile,
        directory=conf_dir,
        input_data=build_input_data(inp),
        pseudopotentials=pseudopotentials,
        kpts=kpts,
    )
    atoms.get_potential_energy()


def main(geom_indexes: list[int] | None):
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    ndata = len(read(inp.system.filename, ":"))
    if geom_indexes is None:
        geom_indexes = list(range(ndata))
    else:
        geom_indexes = [i for i in geom_indexes if i < ndata]  # indexes start from 0

    comm, size, rank, parallel = detect_mpi()
    if parallel:
        check_MPI_tasks_count(comm, len(geom_indexes), "structures")
    my_indexes = distribute_jobs(comm, geom_indexes)

    for geom_idx in my_indexes:
        conf_dir = osp.join(inp.qm.path2qm, f"conf_{geom_idx + 1}")
        if not osp.isdir(conf_dir):
            raise FileNotFoundError(f"{conf_dir} not found; run `python3 -m salted.qe.xyz2sys` first.")
        print(f"[rank {rank}] running pw.x for {conf_dir}", flush=True)
        try:
            run_one(inp, conf_dir)
        except Exception as exc:
            print(f"[rank {rank}] FAILED {conf_dir}: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # create a parser obj, which accepts the indexes to calculate, start from 0
    # formats: 1,2,3 or 1-3 or None (all structures)
    parser.add_argument("-i", "--idx", type=str, default="all", help=ARGHELP_INDEX_STR)
    args = parser.parse_args()

    main(parse_index_str(args.idx))
