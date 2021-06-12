#!/usr/bin/bash
  
cd /local/big_scratch/water_monomer/
mkdir soaps
cd -

for i in 0 1 2 3 4 5
do
   $SALTEDPATH/../SOAPFAST/soapfast/get_power_spectrum.py -f water_monomers_1k.xyz -lm ${i} -c H O -s H O -l 4 -n 5 -o /local/big_scratch/water_monomer/soaps/SOAP-${i}
done
