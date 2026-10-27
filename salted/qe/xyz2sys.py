"""Generate per-structure working directories for Quantum ESPRESSO calculations.

Mirrors `salted.cp2k.xyz2sys`: reads the geometry file (`inp.system.filename`) and creates one
`conf_N` folder per structure under `inp.qm.path2qm`. Unlike the CP2K version -- which only writes
`&COORD`/`&CELL` snippets, since CP2K input decks are hand-assembled by the user via `@INCLUDE` --
each folder here gets a full ASE-readable structure file, since `salted.qe.run_qe` builds the pw.x
input directly from it with ASE's `Espresso` calculator.
"""

import os

from ase.io import read, write

from salted.sys_utils import ParseConfig

STRUCTURE_FNAME = "structure.xyz"


def build():
    inp = ParseConfig().parse_input()
    assert inp.qm.qmcode.lower() == "qe", f"{inp.qm.qmcode=}, but expected 'qe'"

    geoms = read(inp.system.filename, ":")

    for iconf, atoms in enumerate(geoms):
        dirpath = os.path.join(inp.qm.path2qm, f"conf_{iconf + 1}")
        os.makedirs(dirpath, exist_ok=True)
        write(os.path.join(dirpath, STRUCTURE_FNAME), atoms, format="extxyz")

    print(f"Wrote {len(geoms)} structures to {inp.qm.path2qm}/conf_*/{STRUCTURE_FNAME}")


if __name__ == "__main__":
    build()
