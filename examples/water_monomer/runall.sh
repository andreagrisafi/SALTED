#!/usr/bin/bash

python ../../src/sparse_set.py 

python ../../src/kernel_mm.py 

python ../../src/kernel_nm.py 

python ../../src/initialize.py

python ../../src/matrices.py

python ../../src/learn.py 

python ../../src/predict.py 

python ../../src/error.py
