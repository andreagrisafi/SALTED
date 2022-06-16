lam=$(($1))


#RUN TO SELECT nc SPARSE FEATURES FROM ns RANDOM STRUCTURES
$PATH2TENSOAP/get_power_spectrum.py -f coords_1k.xyz -lm ${lam} -p -s H O -c H O -nc 1000 -ns 100 -sm 'random' -o FEAT-${lam}

#RUN TO COMPUTE FEATURES WITH SPARSE FEATURES PRESELECTED
$PATH2TENSOAP/get_power_spectrum.py -f coords_1l.xyz -lm ${lam} -p -s H O -c H O -sf FEAT-${lam} -o FEAT-${lam}
