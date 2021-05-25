#!/usr/bin/bash

#python $SALTEDPATH/ortho_projections.py
python $SALTEDPATH/rkhs_sparse-kernels.py 
python $SALTEDPATH/rkhs_regression.py
python $SALTEDPATH/error.py
