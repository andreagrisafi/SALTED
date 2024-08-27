import os
import sys
import time

from salted import wigner, sparsify_features, scalar_vector
from salted.sys_utils import ParseConfig

def build():

    inp = ParseConfig().parse_input()
    # check for destructive interactions
    if inp.system.average == True and inp.salted.saltedtype == "density-response":
        raise ValueError(
            "Invalid configuration: 'average' cannot be True when 'saltedtype' is 'density-response'. Please change your input settings."
        )

    # Precompute and save the required Wigner-3j symbols and Clebsch-Gordan, depending on SALTED target
    wigner.build()

    # Sparsify the feature space of symmetry-adapted descriptors?
    if inp.descriptor.sparsify.ncut > 0:

        if inp.salted.saltedtype=="density-response":
            print("ERROR: feature space sparsification not allowed with inp.salted.saltedtype: density-response!")
            sys.exit(0)

        # Precompute and save the feature space sparsification details 
        sparsify_features.build()

        # Compute and save the sparsified scalar descriptor 
        scalar_vector.build()

    else:

        # Compute and save the unsparsified scalar descriptor 
        scalar_vector.build()


if __name__ == "__main__":
    build()
