#!/usr/bin/bash

for i in {1..10}; do python $SALTEDPATH/run_pyscf.py -iconf ${i}; done
