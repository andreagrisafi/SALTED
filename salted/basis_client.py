import os
import pickle
import sys
from typing import Dict, List, Optional, Tuple, TypedDict

import yaml


class SpeciesBasisData(TypedDict):
    """Basis data for one chemical species.
    Satisfies: len(nmax) == lmax + 1, and nmax: [s, p, d, ...]
    """
    lmax: int
    nmax: List[int]

def compare_species_basis_data(data1: SpeciesBasisData, data2: SpeciesBasisData):
    """Compare two species basis data.
    If the two species basis data are the same, return True.
    Else, return False.
    """
    return pickle.dumps(data1) == pickle.dumps(data2)

def compare_basis_data_dup_spe(
    basis_data1: Dict[str, SpeciesBasisData],
    basis_data2: Dict[str, SpeciesBasisData],
):
    """Compare two basis data.
    If the SpeciesBasisData for duplicated species are the same, return True.
    Else, return False.
    """
    species_union = set(basis_data1.keys()).intersection(set(basis_data2.keys()))
    for spe_name in species_union:
        # compare by pickle serialization to avoid the problem of comparing lists
        if not compare_species_basis_data(basis_data1[spe_name], basis_data2[spe_name]):
            return False
    return True


class BasisClient:
    """
    Maintain a KSDFT basis data dataset (in yaml format), provide methods to read and write the dataset.
    The class will check the sanity of the dataset file when initialized,
    and check the consistency of the basis data when writing.
    This class will never keep the basis data in memory, but read and write the dataset file every time needed.

    Usage:
    ```python
    basis_client = BasisClient()        # instantiate the basis client
    basis_data = basis_client.read("my_basis")       # read basis data
    lmax, nmax = basis_client.read_as_old_format("my_basis")  # read basis data in the old format (see docstring)
    basis_client.write("my_basis", {"H": {"lmax": 1, "nmax": [4, 3]}, "O": {"lmax": 2, "nmax": [5, 4, 3]}})  # write basis data
    basis_client.pop("my_basis")        # remove basis data
    ```

    YAML structure: (indent with 2 spaces)
    ```yaml
    my_basis:
      H:
        lmax: 1
        nmax: [4, 3]
      O:
        lmax: 2
        nmax: [5, 4, 3]
    ```

    Python dict structure:
    ```python
    {
        "my_basis": {
            "H": {
                "lmax": 1,
                "nmax": [4, 3]
            },
            "O": {
                "lmax": 2,
                "nmax": [5, 4, 3]
            }
        }
    }
    ```
    """

    DEFAULT_DATA_FNAME = "basis_data.yaml"

    def __init__(self, _dev_data_fpath: Optional[str] = None):
        """Initialize the basis client with the data file path.

        Args:
            _dev_data_fpath (optional): For development only!!! Do not use this argument!!!
                The path to the dataset file. If not provided, the default dataset file will be used.
        """
        if _dev_data_fpath is None:
            self.data_fpath = os.path.join(
                self.__salted_package_root, self.DEFAULT_DATA_FNAME
            )
        else:
            print(
                f"[{self.__class__.__name__}] WARNING: _dev_data_fpath is for development only!!!",
                file=sys.stderr,
            )
            self.data_fpath = _dev_data_fpath

        """create the data file if it does not exist"""
        if not os.path.isfile(self.data_fpath):
            print(
                f"Creating density fitting basis dataset file at {self.data_fpath}",
                file=sys.stderr,
            )
            with open(self.data_fpath, "w") as f:
                f.write("")
        self.check_sanity()

    def __repr__(self):
        return f"BasisClient(data_fpath={self.data_fpath})"

    @property
    def __salted_package_root(self) -> str:
        """Get the root directory of the salted package"""
        try:
            import salted
        except ImportError:
            raise ImportError(
                "The salted package is not (properly) installed. Please see https://github.com/andreagrisafi/SALTED"
            )
        assert len(salted.__path__) == 1
        salted_root = salted.__path__[0]
        assert os.path.isdir(salted_root)
        return salted_root

    def check_sanity(self):
        """Check the sanity of the dataset file.

        Current check:
        1. If there are duplicated basis names (by yaml raising error)
        2. lmax nmax consistency
        n. (plz add more checks here if needed)
        """
        with open(self.data_fpath) as f:
            """Checking 1 is done here"""
            basis_data = yaml.safe_load(f)
        if basis_data is None:
            print(f"Empty basis dataset file at {self.data_fpath}", file=sys.stderr)
            return
        basis_names = basis_data.keys()
        for basis_name in basis_names:
            basis_data = self.read(basis_name)
            for spe_name, spe_data in basis_data.items():
                """Checking 2 is done here"""
                assert (
                    spe_data["lmax"] == len(spe_data["nmax"]) - 1
                ), f"lmax nmax discrepancy: {basis_name=}, {spe_name=}, {spe_data=}"

    def _read_all(self) -> Dict[str, Dict[str, SpeciesBasisData]]:
        """Read all basis data from the dataset file"""
        with open(self.data_fpath) as f:
            basis_data_all = yaml.safe_load(f)
        return basis_data_all

    def read(self, basis_name: str) -> Dict[str, SpeciesBasisData]:
        """Read basis data from the dataset file"""
        basis_data_all = self._read_all()
        assert (
            basis_name in basis_data_all.keys()
        ), f"{basis_name=} not found in {self.data_fpath}"
        return basis_data_all[basis_name]

    def read_as_old_format(
        self, basis_name: str
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int]]:
        """Read basis data and return as the old format

        Old format:
        ```python
        lmax = {
            "H": 1,
            "O": 2
        }
        nmax = {
            ("H", 0): 4,
            ("H", 1): 3,
            ("O", 0): 5,
            ("O", 1): 4,
            ("O", 2): 3
        }
        ```
        """
        # print(
        #     "Old format is deprecated and will be removed in the future. Please call read() instead.",
        #     file=sys.stderr,
        # )
        basis_data = self.read(basis_name)
        lmax = {spe_name: spe_data["lmax"] for spe_name, spe_data in basis_data.items()}
        nmax = {
            (spe_name, i): n
            for spe_name, spe_data in basis_data.items()
            for i, n in enumerate(spe_data["nmax"])
        }
        return (lmax, nmax)

    def _write_all(self, basis_data_all: Dict[str, Dict[str, SpeciesBasisData]]):
        """Rewrite the whole dataset file with the new basis data"""
        with open(self.data_fpath, "w") as f:
            yaml.safe_dump(
                basis_data_all, f, default_flow_style=None
            )  # default_flow_style is important!

    def write(self, basis_name: str, basis_data: Dict[str, SpeciesBasisData]):
        """Write basis data to the dataset file"""
        with open(self.data_fpath) as f:
            basis_data_all: Dict = yaml.safe_load(f)

        """cope with duplication"""
        if basis_name in basis_data_all.keys():
            """compare each basis data serialized by pickle"""
            print(f"{basis_name=} already exists in {self.data_fpath}")
            if compare_basis_data_dup_spe(basis_data, basis_data_all[basis_name]):
                print(
                    f"Basis data for the duplicated species are the same. Write data union to file."
                )
                basis_data_all[basis_name].update(basis_data)
            else:
                raise ValueError(
                    f"Basis data for the duplicated species are different.\
                    \n\rCurrent basis data:\n\t{basis_data}\
                    \n\rBasis data in {self.data_fpath}:\n\t{basis_data_all[basis_name]}"
                )
        else:
            basis_data_all[basis_name] = basis_data

        """write basis data to the file"""
        self._write_all(basis_data_all)

    def pop(self, basis_name: str):
        """Pop basis data from the dataset file"""
        basis_data_all = self._read_all()
        if basis_name not in basis_data_all.keys():
            print(
                f"{basis_name=} not found in {self.data_fpath}, no change is made.",
                file=sys.stderr,
            )
        else:
            basis_data_all.pop(basis_name)
            self._write_all(basis_data_all)
            print(f"{basis_name=} has been removed from {self.data_fpath}")


def test_BasisClient():
    """Test the BasisClient class"""

    print("\nTEST: testing BasisClient...")

    print("\nTEST: init with the default dataset file")
    basis_client = BasisClient()

    print("\nTEST: init with a specific dataset file")
    basis_client = BasisClient(
        os.path.join(os.path.dirname(__file__), BasisClient.DEFAULT_DATA_FNAME)
    )

    print("\nTEST: read basis data")
    basis_data = basis_client.read("FHI-aims-light")
    print(basis_data)

    print("\nTEST: read basis data in the old format")
    basis_data_old_format = basis_client.read_as_old_format("FHI-aims-light")
    print(basis_data_old_format)

    print("\nTEST: write basis data")
    test_basis_name = "__I_am_unique__"
    test_basis_data = {
        "H": {"lmax": 1, "nmax": [4, 3]},
        "O": {"lmax": 2, "nmax": [5, 4, 3]},
    }
    basis_client.write(test_basis_name, test_basis_data)

    print("\nTEST: remove basis data")
    basis_client.pop(test_basis_name)

    print("\nTEST: write duplicated basis data")
    assert test_basis_name not in basis_client._read_all().keys()
    basis_client.write(test_basis_name, test_basis_data)

    print("\nTEST: deal with duplicate species but same data")
    test_basis_data1 = test_basis_data.copy()
    test_basis_data1["_Ghost"] = {"lmax": 1, "nmax": [4, 3]}
    print(f"write {test_basis_data1=}")
    basis_client.write(test_basis_name, test_basis_data1)
    print(
        f"Current basis data: {test_basis_name} = {basis_client.read(test_basis_name)}"
    )

    print("\nTEST: deal with duplicate species but different data")
    test_basis_data2 = test_basis_data.copy()
    test_basis_data2["H"]["nmax"] = test_basis_data["H"]["nmax"][
        ::-1
    ]  # make a little change
    try:
        print(f"write {test_basis_data2=}")
        basis_client.write(test_basis_name, test_basis_data2)
    except ValueError:
        print("Duplication error caught, print error info:")
        import traceback

        traceback.print_exc()
    else:
        raise ValueError("Did not catch the duplication error!")
    basis_client.pop(test_basis_name)

    print("\nTEST: test done.")


if __name__ == "__main__":
    basis_data_all = BasisClient()._read_all()

    try:
        test_BasisClient()
    except Exception:
        import traceback

        traceback.print_exc()
        print(f"Restore the original basis data")
        BasisClient()._write_all(basis_data_all)

    # ensure the original basis data is unchanged
    assert pickle.dumps(basis_data_all) == pickle.dumps(
        BasisClient()._read_all()
    ), "The original basis data has been changed!"
