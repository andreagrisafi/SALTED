import os
import sys
import time

from salted import sparse_features, scalar_vector

def build():

    sys.path.insert(0, './')
    import inp

    # salted parameters
    filename = inp.filename
    saltedname = inp.saltedname
    sparsify = inp.sparsify
    rep1 = inp.rep1
    rcut1 = inp.rcut1
    sig1 = inp.sig1
    nrad1 = inp.nrad1
    nang1 = inp.nang1
    neighspe1 = inp.neighspe1
    rep2 = inp.rep2
    rcut2 = inp.rcut2
    sig2 = inp.sig2
    nrad2 = inp.nrad2
    nang2 = inp.nang2
    neighspe2 = inp.neighspe2
    ncut = inp.ncut
    M = inp.Menv
    zeta = inp.z

  
    if sparsify:

        sparse_features.build()    
        scalar_vector.build()  

    else: 
 
        scalar_vector.build()  


if __name__ == "__main__":
    build()
