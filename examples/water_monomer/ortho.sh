#!/usr/bin/bash

python $SALTEDPATH/ortho_projections.py
python $SALTEDPATH/phi-vectors.py 
python $SALTEDPATH/ortho_regression.py
python $SALTEDPATH/ortho_error.py
python $SALTEDPATH/error_electrostatics.py
