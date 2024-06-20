import os
import sys
import time

from salted import sparsify_features, scalar_vector
from salted.sys_utils import ParseConfig

def build():

    inp = ParseConfig().parse_input()

    if inp.descriptor.sparsify.ncut > 0:

        sparsify_features.build()
        scalar_vector.build()

    else:

        scalar_vector.build()


if __name__ == "__main__":
    build()
