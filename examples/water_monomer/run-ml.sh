#!/usr/bin/bash

python $SALTEDPATH/sparse_set.py 
python $SALTEDPATH/kernel_mm.py 
python $SALTEDPATH/kernel_nm.py 
python $SALTEDPATH/initialize.py
python $SALTEDPATH/matrices.py
python $SALTEDPATH/learn.py 
python $SALTEDPATH/validate.py 
python $SALTEDPATH/error_validation.py
python $SALTEDPATH/electrostatics.py
