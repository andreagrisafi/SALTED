#! /bin/bash -l

#SBATCH -o ./predict.out.%j
#SBATCH -e ./predict.err.%j
#SBATCH -J predict
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800

export OMP_NUM_THREADS=1
ulimit -s unlimited

DATADIR='./data'
AIMS=~/aims.master.mkl.x

python -m salted.aims.move_data_in -vl

VAL_SET_FILE="temp_validation_set.txt"

mapfile -t structures < $VAL_SET_FILE

for j in ${structures[@]}; do
        let i=$j+1
	cp control_read.in ${DATADIR}/$i/control.in
done

wait 

for j in ${structures[@]}; do
        let i=$j+1
	cd ${DATADIR}/$i
	mv ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	srun --exclusive -n 1 $AIMS < /dev/null > aims_predict.out && mv rho_rebuilt_ri.out rho_ml.out && mv ri_restart_coeffs.out ri_restart_coeffs_ml.out &

	cd -
done

wait

rm $VAL_SET_FILE
