#!/usr/bin/bash

python $SALTEDPATH/ortho_projections.py
python $SALTEDPATH/kernels.py 
python $SALTEDPATH/unstable_ortho_regression.py
python $SALTEDPATH/error.py
