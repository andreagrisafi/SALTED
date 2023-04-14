#! /bin/bash -l

#SBATCH -o ./validate.out.%j
#SBATCH -e ./validate.err.%j
#SBATCH -J validate
#SBATCH --partition=XXXX
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800

export OMP_NUM_THREADS=1
ulimit -s unlimited

QMDIR='qmdata/'
AIMS=~/aims.master.mkl.x

DATADIR=${QMDIR}data

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	let j=$i-1
	if grep -Fqx $j training_set.txt; then continue; fi
	cp control_read.in ${DATADIR}/$i/control.in
	python move_data_in.py -i $i
	cd ${DATADIR}/$i
	
	mv ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	srun --exclusive -n 1 $AIMS < /dev/null > aims_validate.out && mv rho_rebuilt_ri.out rho_ml.out && mv ri_restart_coeffs.out ri_restart_coeffs_ml.out &

	cd -
done
