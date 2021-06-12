#!/bin/bash

nsample=$(ls | grep -c KERNEL)
for i in $(seq 1 ${nsample});do
	sagpr_train -r ${1} -reg ${2} -f FRAMES.${i}.xyz -p ALPHA_L2 -sp -perat -w WEIGHTS.${i} -sf KERNEL.${i}.npy ${3} -m pinv
	echo "Trained model number "${i}
done
rm FRAMES.*.xyz KERNEL.*.npy
