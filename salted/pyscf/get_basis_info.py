"""Translate basis info from PySCF calculation to SALTED basis info"""

import os
from typing import Dict, List

import inp
import yaml
from ase.io import read

from pyscf import df
from pyscf.gto import basis

from salted.basis_client import (
    BasisClient,
    SpeciesBasisData,
    compare_species_basis_data,
)

from salted.get_basis_info import get_parser


def build(dryrun: bool = False, force_overwrite: bool = False):
    """Scheme: load density fitting basis from pyscf module,
    update the basis_data dict,
    and write to the database when all species are recorded.
    """
    assert inp.qmcode.lower() == "pyscf", f"{inp.qmcode=}, but expected 'pyscf'"

    spe_set = set(inp.species)  # remove duplicates
    qmbasis = inp.qmbasis

    """load density fitting basis from pyscf module"""
    basis_data: Dict[str, SpeciesBasisData] = load_from_pyscf(list(spe_set), qmbasis)

    """write to the database and working directory"""
    if dryrun:
        print("Dryrun mode, not writing to the database")
        print(f"{basis_data=}")
    else:
        BasisClient().write(inp.dfbasis, basis_data, force_overwrite)
        with open(os.path.join(inp.saltedpath, "new_basis_entry.yaml"), "w") as f:
            yaml.safe_dump(basis_data, f, default_flow_style=None)



def load_from_pyscf(species_list: List[str], qmbasis: str):
    """load the xxx-jkfit density fitting basis from PySCF

    Args:
        species_list: list of species, e.g. [H, O]
        qmbasis: quantum chemistry basis set name, e.g. cc-pvdz

    Returns:
        Dict[str, SpeciesBasisData]: species and basis data
    """
    ribasis = df.addons.DEFAULT_AUXBASIS[basis._format_basis_name(qmbasis)][0]  # get the proper DF basis name in PySCF
    print(f"{species_list=}, {qmbasis=}, and the parsed {ribasis=}")
    spe_ribasis_info = {spe: basis.load(ribasis, spe) for spe in species_list}  # load with PySCF basis module
    """
    Each dict value is like:
        format: [angular_momentum, [exponents, coefficients]]
    [[0, [1113.9867719, 1.0]],
     [0, [48.12711454, 1.0]],
     [0, [2.50686566, 1.0]],
     [0, [0.1918516, 1.0]],
     [1, [102.99176249, 1.0]],
     [1, [3.3490545, 1.0]],
     [1, [0.20320063, 1.0]],
     [2, [10.59406836, 1.0]],
     [2, [0.51949765, 1.0]],
     ...]

    Extract the l numbers and compose the Dict[str, SpeciesBasisData] (species and basis data)
    """
    basis_data = {spe: collect_l_nums(ribasis_info) for spe, ribasis_info in spe_ribasis_info.items()}
    return basis_data


# def collect_l_nums(data:List[int, List[float]]) -> SpeciesBasisData:
# use Annotated
def collect_l_nums(data: List) -> SpeciesBasisData:
    """collect l numbers for each species based on the data from PySCF
    input: above dict value,
        e.g. [[0, [1113.9867719, 1.0]], [1, [102.99176249, 1.0]], ...]
    output: max l number, and a list of counts of each l number
    """
    l_nums = [d for d, _ in data]  # [0, 0, 0, 0, 1, 1, 1, 2, 2, ...]
    l_max = max(l_nums)
    l_cnt = [0 for _ in range(l_max + 1)]  # [0, 0, 0, ...] to the max l number
    for l_num in l_nums:
        l_cnt[l_num] += 1
    return {
        "lmax": max(l_nums),
        "nmax": l_cnt,
    }



if __name__ == "__main__":
    print("Please call `python -m salted.get_basis_info` instead of this file")

    parser = get_parser()
    args = parser.parse_args()

    build(dryrun=args.dryrun, force_overwrite=args.force_overwrite)

