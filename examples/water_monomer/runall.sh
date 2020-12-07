#!/usr/bin/bash

python $RHOMLPATH/sparse_set.py 
python $RHOMLPATH/kernel_mm.py 
python $RHOMLPATH/kernel_nm.py 
python $RHOMLPATH/initialize.py
python $RHOMLPATH/matrices.py
python $RHOMLPATH/learn.py 
python $RHOMLPATH/predict.py 
python $RHOMLPATH/error.py
