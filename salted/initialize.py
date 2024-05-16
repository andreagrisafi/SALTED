import os
import sys
import time

from salted import sparsify_features, scalar_vector
from salted.sys_utils import ParseConfig

def build():
    inp = ParseConfig().parse_input()

    # salted parameters
    (saltedname, saltedpath,
    filename, species, average, field, parallel,
    path2qm, qmcode, qmbasis, dfbasis,
    filename_pred, predname, predict_data,
    rep1, rcut1, sig1, nrad1, nang1, neighspe1,
    rep2, rcut2, sig2, nrad2, nang2, neighspe2,
    sparsify, nsamples, ncut,
    zeta, Menv, Ntrain, trainfrac, regul, eigcut,
    gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()


    if inp.descriptor.sparsify.ncut > 0:

        sparsify_features.build()
        scalar_vector.build()

    else:

        scalar_vector.build()


if __name__ == "__main__":
    build()
