#!/usr/bin/bash
  
source ~/miniconda/bin/activate rascal
for i in 0 1 2 3 4 5
do
   /local/scratch/source/SOAPFAST/soapfast/get_power_spectrum.py -f coords_1000.xyz -lm ${i} -o soaps/SOAP-${i}
done
source ~/miniconda/bin/deactivate rascal
