import logging
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple, TypedDict, Union

import yaml


def compare_by_pickle(obj1: Any, obj2: Any) -> bool:
    if pickle.dumps(obj1) == pickle.dumps(obj2):
        return True
    else:
        return False


class BasisSpeciesData(TypedDict):
    lmax: int
    nmax: List[int]


class BasisClient:
    """
    Maintain a KSDFT basis data dataset (in yaml format), provide methods to read and write the dataset.
    This class will not keep the basis data in memory, but read and write the dataset file every time needed.

    Usage:
    ```python
    basis_client = BasisClient()        # use the default dataset file
    basis_client = BasisClient("path/to/basis_data.yaml")  # use a specific dataset file
    basis_data = basis_client.read("my_basis")       # read basis data
    lmax, nmax = basis_client.read_as_old_format("my_basis")  # read basis data in the old format
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

    def __init__(self, data_fpath: Union[None, str] = None):
        """Initialize the basis client with the data file path.

        Args:
            data_fpath (optional): The path to the dataset file. If not provided, the default dataset file will be used.
        """
        if data_fpath is None:
            self.data_fpath = os.path.join(
                self.__salted_package_root, self.DEFAULT_DATA_FNAME
            )
        else:
            self.data_fpath = data_fpath

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
            raise ImportError("The salted package is not installed.")
        assert len(salted.__path__) == 1
        salted_root = salted.__path__[0]
        assert os.path.isdir(salted_root)
        return salted_root

    def check_sanity(self):
        """Check the sanity of the dataset file.

        Current check:
        - lmax nmax consistency
        - (plz add more checks here if needed)
        """
        with open(self.data_fpath) as f:
            basis_data = yaml.safe_load(f)
        if basis_data is None:
            print(f"Empty basis dataset file at {self.data_fpath}", file=sys.stderr)
            return
        basis_names = basis_data.keys()
        for basis_name in basis_names:
            basis_data = self.read(basis_name)
            for spe_name, spe_data in basis_data.items():
                assert (
                    spe_data["lmax"] == len(spe_data["nmax"]) - 1
                ), f"lmax nmax discrepancy: {basis_name=}, {spe_name=}, {spe_data=}"

    def read(self, basis_name: str) -> Dict[str, BasisSpeciesData]:
        """read basis data from the dataset file"""
        with open(self.data_fpath) as f:
            basis_data_all = yaml.safe_load(f)
        assert (
            basis_name in basis_data_all.keys()
        ), f"{basis_name=} not found in {self.data_fpath}"
        return basis_data_all[basis_name]

    def read_as_old_format(
        self, basis_name: str
    ) -> Tuple[Dict[str, int], Dict[Tuple[str, int], int]]:
        """read basis data and return as the old format

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

    def write(self, basis_name: str, basis_data: Dict[str, BasisSpeciesData]):
        """write basis data to the dataset file"""
        with open(self.data_fpath) as f:
            basis_data_all: Dict = yaml.safe_load(f)

        """cope with duplication"""
        if basis_name in basis_data_all.keys():
            """compare each basis data serialized by pickle"""
            if compare_by_pickle(basis_data_all[basis_name], basis_data):
                print(
                    f"{basis_name=} already exists in {self.data_fpath}, no change is made."
                )
                return
            else:
                raise ValueError(
                    f"{basis_name=} already exists in {self.data_fpath}, but with different content.\
                    \rCurrent basis data:\n\t{basis_data}\
                    \rBasis data in {self.data_fpath}:\n\t{basis_data_all[basis_name]}"
                )

        """write basis data to the file"""
        basis_data_all.update({basis_name: basis_data})
        with open(self.data_fpath, "w") as f:
            yaml.safe_dump(basis_data_all, f, default_flow_style=None)

    def pop(self, basis_name: str):
        """pop basis data from the dataset file"""
        with open(self.data_fpath) as f:
            basis_data_all = yaml.safe_load(f)
        if basis_name not in basis_data_all.keys():
            print(
                f"{basis_name=} not found in {self.data_fpath}, no change is made.",
                file=sys.stderr,
            )
        else:
            basis_data_all.pop(basis_name)
            with open(self.data_fpath, "w") as f:
                yaml.safe_dump(basis_data_all, f, default_flow_style=None)
            print(f"{basis_name=} has been removed from {self.data_fpath}")


def test_BasisClient():
    """test the BasisClient class"""
    print("TEST: testing BasisClient...")

    print("TEST: init with the default dataset file")
    basis_client = BasisClient()

    print("TEST: init with a specific dataset file")
    basis_client = BasisClient(
        os.path.join(os.path.dirname(__file__), BasisClient.DEFAULT_DATA_FNAME)
    )

    print("TEST: read basis data")
    basis_data = basis_client.read("FHI-aims-light")
    print(basis_data)

    print("TEST: read basis data in the old format")
    basis_data_old_format = basis_client.read_as_old_format("FHI-aims-light")
    print(basis_data_old_format)

    print("TEST: write duplicated basis data")
    basis_client.write("FHI-aims-light", basis_data)

    print("TEST: try to catch same-name duplication error")
    basis_data_fake = basis_data.copy()
    basis_data_fake["Ghost"] = {"lmax": 1, "nmax": [4, 3]}
    try:
        basis_client.write("FHI-aims-light", basis_data_fake)
    except ValueError as e:
        print("Duplication error caught, print error info:")
        print(e)
    else:
        raise ValueError("Did not catch the duplication error!")

    print("TEST: write basis data")
    basis_client.write("FHI-aims-light_copy", basis_data)

    print("TEST: remove basis data")
    basis_client.pop("FHI-aims-light_copy")

    print("TEST: test done.")


if __name__ == "__main__":
    test_BasisClient()
