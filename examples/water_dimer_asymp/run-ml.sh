#!/usr/bin/bash

python $SALTEDPATH/kernel_tm.py
python $SALTEDPATH/predict.py 
python $SALTEDPATH/error_prediction.py
