#!/usr/bin/bash

for i in {751..1000}; do python $RHOMLPATH/dm2df.py -iconf ${i}; done
