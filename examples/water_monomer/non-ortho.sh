#!/usr/bin/bash

python $SALTEDPATH/phi-vectors.py
python $SALTEDPATH/phi-matrices.py
python $SALTEDPATH/regression.py
python $SALTEDPATH/error_electrostatics.py
