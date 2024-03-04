"""translate basis info from aims calculation to SALTED basis info"""

import argparse
import os
from typing import Dict, List, Tuple

import inp
import yaml
from ase.io import read

from salted.basis_client import (
    BasisClient,
    SpeciesBasisData,
    compare_species_basis_data,
)


def build(dryrun: bool = False):
    """Scheme: parse all basis_info.out one by one,
    update the basis_data dict,
    and write to the database when all species are recorded.
    """
    spe_set = set(inp.species)
    geoms_list = read(inp.filename, ":")
    qmdata_dpath = os.path.join(inp.path2qm, "data")
    basis_data: Dict[str, SpeciesBasisData] = {}  # hold all species basis data

    for iconf, geom in enumerate(geoms_list):
        """parse basis_info.out for this geometry"""
        basis_info_fpath = os.path.join(qmdata_dpath, str(iconf + 1), "basis_info.out")
        spe_basis_data_list = parse_file_basis_info(basis_info_fpath)
        chem_syms = geom.get_chemical_symbols()
        assert len(chem_syms) == len(
            spe_basis_data_list
        ), f"{len(chem_syms)=}, {len(spe_basis_data_list)=}, {basis_info_fpath=}"
        chem_syms_uniq_idxes = [chem_syms.index(spe) for spe in set(chem_syms)]
        chem_syms_uniq = [chem_syms[i] for i in chem_syms_uniq_idxes]
        spe_basis_data_list_uniq = [
            spe_basis_data_list[i] for i in chem_syms_uniq_idxes
        ]

        """update/compare basis data for each species"""
        for spe, spe_basis_data in zip(
            chem_syms_uniq, spe_basis_data_list_uniq, strict=True
        ):
            if spe not in basis_data.keys():
                basis_data[spe] = spe_basis_data
            else:
                if not compare_species_basis_data(basis_data[spe], spe_basis_data):
                    raise ValueError(
                        f"Species {spe} has inconsistent basis data: {basis_data[spe]} and {spe_basis_data}, file: {basis_info_fpath}"
                    )

        """check if all species are recorded"""
        if set(basis_data.keys()) == spe_set:
            break

    """double check if all species are recorded"""
    if set(basis_data.keys()) != spe_set:
        raise ValueError(
            f"Not all species are recorded: {basis_data.keys()} vs {spe_set}"
        )

    """write to the database"""
    if dryrun:
        print("Dryrun mode, not writing to the database")
        print(f"{basis_data=}")
    else:
        BasisClient().write(inp.dfbasis, basis_data)
        with open(os.path.join(inp.saltedpath, "new_basis_entry.yaml"), "w") as f:
            yaml.safe_dump(basis_data, f, default_flow_style=None)


def parse_file_basis_info(basis_info_fpath: str) -> List[SpeciesBasisData]:
    """Parse basis_info.out by FHI-aims.
    Parse by str.strip().split().

    File example: (might have multiple atoms, has a space at the beginning of each line)
    ```text
     atom           1
     For L =            0 there are           9 radial functions
     For L =            1 there are          10 radial functions
     For L =            2 there are           9 radial functions
     For L =            3 there are           8 radial functions
     For L =            4 there are           6 radial functions
     For L =            5 there are           4 radial functions
     For atom            1 max L =            5
    ```

    Return: in the format of List[SpeciesBasisData]
    ```python
    [
        {
            "lmax": 5,
            "nmax": [9, 10, 9, 8, 6, 4],
        },
        ...  # other atoms
    ]
    ```
    """

    with open(basis_info_fpath) as f:
        lines_raw = f.readlines()
    lines_raw = [i.strip().split() for i in lines_raw]

    """Step 1: further split the current list by the keyword `atom`"""
    boundary_1atom = [
        idx for idx, line in enumerate(lines_raw) if line[0] == "atom"
    ] + [len(lines_raw)]
    lines_by_atoms = [
        lines_raw[boundary_1atom[i] : boundary_1atom[i + 1]]
        for i in range(len(boundary_1atom) - 1)
    ]
    # if dryrun:
    #     print(f"{boundary_1atom=}")
    #     print(f"{lines_by_atoms=}")

    """Step 2: derive SpeciesBasisData and do some checks"""
    basis_data: List[SpeciesBasisData] = []
    for lines_atom in lines_by_atoms:
        err_msg = f"{basis_info_fpath=}, {lines_atom=}"
        assert lines_atom[0][0] == "atom", err_msg  # check the first line
        assert lines_atom[-1][0] == "For", err_msg  # check the last line
        assert (
            len(lines_atom) == int(lines_atom[-1][-1]) + 3
        ), err_msg  # check the number of lines
        basis_spe_data = {
            "lmax": int(lines_atom[-1][-1]),
            "nmax": [int(i[6]) for i in lines_atom[1:-1]],
        }
        assert (
            len(basis_spe_data["nmax"]) == basis_spe_data["lmax"] + 1
        ), f"{basis_info_fpath=}, {basis_spe_data=}"
        basis_data.append(basis_spe_data)

    return basis_data


def write_basis_info_old(
    l_list: Dict[str, int], n_list: Dict[Tuple[str, int], int], dryrun: bool = False
):
    l_list = {k: v - 1 for k, v in l_list.items()}
    basis_name = inp.dfbasis
    basis_data: Dict[str, SpeciesBasisData] = {}
    for spe in l_list.keys():
        assert l_list[spe] + 1 == len(
            [i for i in n_list.keys() if i[0] == spe]
        )  # compare l_list with n_list
        basis_data[spe] = {
            "lmax": l_list[spe],
            "nmax": [n_list[(spe, l)] for l in range(l_list[spe] + 1)],
        }
    if dryrun:
        print("Dryrun mode, not writing to the database")
        print(f"{inp.species=}")
        print(f"{l_list=}, {n_list=}")
        print(f"inp.dfbasis={basis_name=}")
        print(f"{basis_data=}")
    else:
        basis_client = BasisClient()
        basis_client.write(basis_name, basis_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="run without writing to the database, print the result",
    )

    args = parser.parse_args()

    build(dryrun=args.dryrun)