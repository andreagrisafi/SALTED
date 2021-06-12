#!/bin/nbash

mkdir $1;cd $1

# build L=0 power spectrum
sagpr_get_PS -f ../coords.xyz -lm 0 -n 3 -l 3 -o PS0 > /dev/null

# build L=2 power spectrum
sagpr_get_PS -f ../coords.xyz -lm 2 -n 3 -l 3 -o PS2 > /dev/null

# get atomic power spectra and do environmental sparsification
get_atomic_power_spectrum.py -p PS0.npy -f ../coords.xyz -o PS0_atomic > /dev/null
get_atomic_power_spectrum.py -p PS2.npy -f ../coords.xyz -o PS2_atomic > /dev/null
sagpr_do_env_fps -p PS0_atomic.npy -n 200 -o PS0_atomic_sparse -i 0 > /dev/null
sagpr_do_env_fps -p PS2_atomic.npy -n 200 -o PS2_atomic_sparse -i 0 > /dev/null
sagpr_apply_env_fps -p PS0_atomic.npy -sf PS0_atomic_sparse_rows -o PS0_atomic_sparse > /dev/null
sagpr_apply_env_fps -p PS2_atomic.npy -sf PS2_atomic_sparse_rows -o PS2_atomic_sparse > /dev/null

# build L=0 kernels
sagpr_get_kernel -z 2 -ps PS0.npy PS0_atomic_sparse.npy -o KER0_NM -s PS0_natoms.npy NONE > /dev/null
sagpr_get_kernel -z 2 -ps PS0_atomic_sparse.npy -o KER0_MM -s NONE > /dev/null

# build L=2 kernels
sagpr_get_kernel -z 2 -ps PS2.npy PS2_atomic_sparse.npy -ps0 PS0.npy PS0_atomic_sparse.npy -o KER2_NM -s PS2_natoms.npy NONE > /dev/null
sagpr_get_kernel -z 2 -ps PS2_atomic_sparse.npy -ps0 PS0_atomic_sparse.npy -o KER2_MM -s NONE > /dev/null

# do cartesian regression
sagpr_train -r 2 -reg 1e-9 1e-6 -f ../coords.xyz -p alpha -sel 0 10 -sf KER0_NM.npy KER0_MM.npy KER2_NM.npy KER2_MM.npy -perat -w wt_cart -m pinv > /dev/null

# do spherical regression
sagpr_cart_to_sphr -f ../coords.xyz -o coords.xyz -p alpha -r 2 > /dev/null
sagpr_train -r 0 -sp -reg 1e-9 -f coords.xyz -p alpha_L0 -sel 0 10 -sf KER0_NM.npy KER0_MM.npy -perat -w wt_sphr -m pinv > /dev/null
sagpr_train -r 2 -sp -reg 1e-6 -f coords.xyz -p alpha_L2 -sel 0 10 -sf KER2_NM.npy KER2_MM.npy -perat -w wt_sphr -m pinv > /dev/null

echo "CHECK DIFFERENCES IN WEIGHTS"
python -c 'import numpy as np;wC = np.load("wt_cart_0.npy",allow_pickle=True);wS = np.load("wt_sphr_0.npy",allow_pickle=True);print (100 * np.linalg.norm(wC[4]-wS[4]) / np.linalg.norm(wC[4]),"%")'
python -c 'import numpy as np;wC = np.load("wt_cart_2.npy",allow_pickle=True);wS = np.load("wt_sphr_2.npy",allow_pickle=True);print (100 * np.linalg.norm(wC[4]-wS[4]) / np.linalg.norm(wC[4]),"%")'

cd ../
