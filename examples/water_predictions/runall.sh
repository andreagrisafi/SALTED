#!/usr/bin/bash

python $RHOMLPATH/sparse_set.py 
python $RHOMLPATH/kernel_mm.py 
python $RHOMLPATH/kernel_nm.py 
python $RHOMLPATH/initialize.py
python $RHOMLPATH/matrices.py
python $RHOMLPATH/learn.py 

# OUT-OF-SAMPLE PREDICTION
python $RHOMLPATH/kernel_tm.py 
python $RHOMLPATH/predict.py 
python $RHOMLPATH/error_prediction.py

# IN-SAMPLE VALIDATION
#python $RHOMLPATH/validate.py 
#python $RHOMLPATH/error_validation.py
