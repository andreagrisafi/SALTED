"""Extract converged Quantum ESPRESSO electron densities as Gaussian cube files.

Runs `pp.x` on each `conf_N` folder produced by `salted.qe.run_qe` (`plot_num = 0`: valence
electron density) and writes a Gaussian cube file (`CUBE_FNAME`) that the rest of SALTED's
cube-reading machinery -- and the density-fitting step that eventually consumes it -- can read
like any other cube file.

Note: like CP2K, this is a *valence* electron density (QE uses pseudopotentials too). Integrating
the cube reproduces the sum of each species' UPF `Z_valence`, not the sum of atomic numbers.
"""

import argparse
import os.path as osp
import subprocess

from ase.io import read

from salted.qe.run_qe import PWSCF_OUTDIR, PWSCF_PREFIX, default_if_placeholder
from salted.sys_utils import ARGHELP_INDEX_STR, ParseConfig, parse_index_str

CUBE_FNAME = "rho.cube"
PP_INPUT_FNAME = "pp.in"
PP_OUTPUT_FNAME = "pp.out"

PP_INPUT_TEMPLATE = """\
&INPUTPP
  prefix = '{prefix}'
  outdir = '{outdir}'
  plot_num = 0
/
&PLOT
  iflag = 3
  output_format = 6
  fileout = '{fileout}'
/
"""


def pp_command(inp) -> list[str]:
    ppx = default_if_placeholder(inp.qm.ppx_command, "pp.x")
    nproc = default_if_placeholder(inp.qm.nproc, 1)
    if nproc > 1:
        return ["mpirun", "-np", str(nproc), *ppx.split()]
    return ppx.split()


def run_one(inp, conf_dir: str):
    """Run pp.x inside `conf_dir`, extracting its SCF density to a Gaussian cube file."""
    pp_input = PP_INPUT_TEMPLATE.format(prefix=PWSCF_PREFIX, outdir=PWSCF_OUTDIR, fileout=CUBE_FNAME)
    with open(osp.join(conf_dir, PP_INPUT_FNAME), "w") as f:
        f.write(pp_input)

    with open(osp.join(conf_dir, PP_OUTPUT_FNAME), "w") as f:
        subprocess.run(
            [*pp_command(inp), "-in", PP_INPUT_FNAME],
            cwd=conf_dir,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main(geom_indexes: list[int] | None):
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    ndata = len(read(inp.system.filename, ":"))
    if geom_indexes is None:
        geom_indexes = list(range(ndata))
    else:
        geom_indexes = [i for i in geom_indexes if i < ndata]  # indexes start from 0

    # Post-processing is cheap relative to the SCF step, so this is intentionally run serially;
    # unlike salted.qe.run_qe it does not distribute conf_N folders across MPI ranks.
    for geom_idx in geom_indexes:
        conf_dir = osp.join(inp.qm.path2qm, f"conf_{geom_idx + 1}")
        if not osp.isdir(conf_dir):
            raise FileNotFoundError(f"{conf_dir} not found; run `python3 -m salted.qe.xyz2sys` first.")
        print(f"running pp.x for {conf_dir}", flush=True)
        try:
            run_one(inp, conf_dir)
        except Exception as exc:
            print(f"FAILED {conf_dir}: {exc}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--idx", type=str, default="all", help=ARGHELP_INDEX_STR)
    args = parser.parse_args()

    main(parse_index_str(args.idx))
