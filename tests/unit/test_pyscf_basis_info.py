"""Unit tests for salted.pyscf.get_basis_info (requires pyscf; skipped otherwise).

Guards the RI-cc-pvqz entry shipped in SALTED-datasets/water_monomer_PySCF/
basis_data.yaml against drift: the entry there was generated with exactly this
code path.
"""

import pytest

pyscf = pytest.importorskip("pyscf")

from salted.pyscf.get_basis_info import collect_l_nums, load_from_pyscf  # noqa: E402

# reference: what `python -m salted.get_basis_info --dryrun` produces for the
# water_monomer_PySCF example (qmbasis=cc-pvqz -> ribasis=cc-pvqz-jkfit)
RI_CC_PVQZ = {
    "H": {"lmax": 4, "nmax": [4, 3, 3, 2, 1]},
    "O": {"lmax": 5, "nmax": [10, 7, 5, 3, 2, 1]},
}


def test_load_from_pyscf_water_ri_cc_pvqz():
    basis_data = load_from_pyscf(["H", "O"], "cc-pvqz")
    assert basis_data == RI_CC_PVQZ


def test_collect_l_nums():
    # two s shells, one p shell, one d shell
    data = [
        [0, [883.99, 0.33], [286.84, 0.81]],
        [0, [48.13, 1.0]],
        [1, [102.99, 1.0]],
        [2, [10.59, 1.0]],
    ]
    assert collect_l_nums(data) == {"lmax": 2, "nmax": [2, 1, 1]}
