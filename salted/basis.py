from salted.basis_client import BasisClient


def basiset(basis: str, dfbasis_file: str | None = None):
    """read basis data and return as the old format

    WARNING: Please use BasisClient() to read basis data instead of this function.
        See BasisClient docstring for more information.

    Args:
        basis: name of the density-fitting basis to read (inp.qm.dfbasis).
        dfbasis_file: optional path to an external basis dataset file
            (inp.qm.dfbasis_file); if None, the default package basis is used.

    Return:
    (lmax, nmax), using the old format

    Old format:
    ```python
    lmax = {
       "H": 1,
       "O": 2,
    }
    nmax = {
       ("H", 0): 4,
       ("H", 1): 3,
       ("O", 0): 5,
       ("O", 1): 4,
       ("O", 2): 3,
    }
    ```
    """
    return BasisClient(data_fpath=dfbasis_file).read(basis)
