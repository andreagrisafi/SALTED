from enum import Enum
from typing import Union

class CutoffType(str, Enum):
    ESTIMATE = "estimate"
    NON_PERIODIC = "non-periodic"
    FIRST_NEIGHBOURS = "first-neighbours"

Cutoff = Union[float, CutoffType]

