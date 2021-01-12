#!/usr/bin/bash

for i in {1..10}; do python $SALTEDPATH/dm2df.py -iconf ${i}; done
