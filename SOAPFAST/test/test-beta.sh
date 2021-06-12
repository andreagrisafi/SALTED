#!/bin/nbash

mkdir $1;cd $1

# build L=0 power spectrum
sagpr_get_PS -f ../coords_water.xyz -lm 0 -n 3 -l 3 -o PS0 > /dev/null

# build L=1 power spectrum
sagpr_get_PS -f ../coords_water.xyz -lm 1 -n 3 -l 3 -o PS1 > /dev/null

# build L=3 power spectrum
sagpr_get_PS -f ../coords_water.xyz -lm 3 -n 2 -l 2 -o PS3 > /dev/null

# build L=1 kernel
sagpr_get_kernel -z 2 -ps PS1.npy -ps0 PS0.npy -o KER1 -s PS1_natoms.npy > /dev/null

# build L=3 kernel
sagpr_get_kernel -z 2 -ps PS3.npy -ps0 PS0.npy -o KER3 -s PS3_natoms.npy > /dev/null

# do cartesian regression
sagpr_train -r 3 -reg 1e-7 1e-5 -f ../coords_water.xyz -p beta -sel 0 5 -w wt_cart -perat -pr -k KER1.npy KER3.npy | tee regression.out > /dev/null

# do spherical regression
sagpr_cart_to_sphr -f ../coords_water.xyz -o coords.xyz -p beta -r 3 > /dev/null
sagpr_train -r 01 -sp -reg 1e-7 -f coords.xyz -p beta_L01 -sel 0 5 -w wt_sphr -perat -pr -k KER1.npy | tee regression_L01.out > /dev/null
sagpr_train -r 21 -sp -reg 1e-7 -f coords.xyz -p beta_L21 -sel 0 5 -w wt_sphr -perat -pr -k KER1.npy | tee regression_L21.out > /dev/null
sagpr_train -r 23 -sp -reg 1e-5 -f coords.xyz -p beta_L23 -sel 0 5 -w wt_sphr -perat -pr -k KER3.npy | tee regression_L23.out > /dev/null

# do cartesian prediction
python -c 'import numpy as np;k1 = np.load("KER1.npy");k3 = np.load("KER3.npy");np.save("KER1_TT.npy",k1[:,:5]);np.save("KER3_TT.npy",k3[:,:5])'
sagpr_prediction -w wt_cart -r 3 -k KER1_TT.npy KER3_TT.npy -o prediction > /dev/null

# do spherical prediction
sagpr_prediction -w wt_sphr -r 01 -sp -k KER1_TT.npy -o prediction > /dev/null
sagpr_prediction -w wt_sphr -r 21 -sp -k KER1_TT.npy -o prediction > /dev/null
sagpr_prediction -w wt_sphr -r 23 -sp -k KER3_TT.npy -o prediction > /dev/null

echo "CHECK DIFFERENCES IN WEIGHTS"
python -c 'import numpy as np;wC = np.load("wt_cart_01.npy",allow_pickle=True);wS = np.load("wt_sphr_01.npy",allow_pickle=True);print (100 * np.linalg.norm(wC[4]-wS[4]) / np.linalg.norm(wC[4]),"%")'
python -c 'import numpy as np;wC = np.load("wt_cart_23.npy",allow_pickle=True);wS = np.load("wt_sphr_23.npy",allow_pickle=True);print (100 * np.linalg.norm(wC[4]-wS[4]) / np.linalg.norm(wC[4]),"%")'

cd ../
