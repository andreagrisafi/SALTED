#!/bin/bash

# This script is used for making predictions on a validation set using a committee of models. It is currently hard-coded to use the lambda=2 property alpha_L2, but more detail is given in the following comments.

# Get number of models
nsample=$(ls | grep -c WEIGHTS)

for i in $(seq 1 ${nsample});do
	# Get the prediction for this model (assuming the kernel between members of the validation set and the active set are stored in ../K2_validation.npy).
	sagpr_prediction -r 2 -w WEIGHTS.${i} -k ../K2_validation.npy -o PREDICTION.${i} -sp
	# Get calculated values for comparison. Here we are searching for an order-2 spherical tensor called alpha_L2, and writing it to the CALC.*_L2.txt files.
	# This could also be done with python and ASE, but the awk method is faster.
	cat ../validation.xyz | sed "s/\(=\|\"\)/ /g" | awk '{if (NF==1){nat=$1}}/Properties/{for (i=1;i<=NF;i++){if ($i=="alpha_L2"){printf "%.16f %.16f %.16f %.16f %.16f\n", $(i+1)/nat,$(i+2)/nat,$(i+3)/nat,$(i+4)/nat,$(i+5)/nat}}}' > CALC.${i}_L2.txt
	# Combine predicted and calculated values and get residuals. The way to do this depends on the property you want to predict, its rank and how you define the error.
	# e.g., here, we have element-wise residuals because our error of choice is the RMSE.
	# If instead we wanted to do the MAE for dipole moments, we might want to put a single line into the residuals file for each calculation,
	# the predicted norm followed by the calculated one.
	paste PREDICTION.${i}_L2.txt CALC.${i}_L2.txt | awk '{print $1,$6;print $2,$7;print $3,$8;print $4,$9;print $5,$10}' > RESIDUAL.${i}.txt
	echo "Predicted model number "${i}
done
