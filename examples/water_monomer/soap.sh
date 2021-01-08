#!/usr/bin/bash
  
source ~/miniconda/bin/activate rascal
for i in 0 1 2 3 4 5
do
   /local/scratch/source/SOAPFAST/soapfast/get_power_spectrum.py -f coords_1000.xyz -c H O -s H O -rc 3.0 -sg 0.3 -n 3 -l 2 -lm ${i} -o soaps/SOAP-${i}
done
source ~/miniconda/bin/deactivate rascal
