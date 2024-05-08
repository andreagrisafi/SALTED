from salted.basis_client import BasisClient


def basiset(basis: str):
    """read basis data and return as the old format

    WARNING: Please use BasisClient() to read basis data instead of this function.
        See BasisClient docstring for more information.

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
    return BasisClient().read_as_old_format(basis)  # use default basis data file
