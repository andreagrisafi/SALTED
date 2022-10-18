#! /bin/bash -l

#SBATCH -o ./predict.out.%j
#SBATCH -e ./predict.err.%j
#SBATCH -J predict
#SBATCH --partition=mpsd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800

export OMP_NUM_THREADS=1
ulimit -s unlimited

DATADIR='qmdata/predicted_data'
AIMS=~/aims.master.mkl.x

python make_geoms.py --predict

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir ${DATADIR}/$i
	cp control_predict.in ${DATADIR}/$i/control.in
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	cd ${DATADIR}/$i

	srun -n 1 $AIMS < /dev/null > aims.out &

	cd -
done

wait

for (( i=1; i<=$n; i++ )); do
	cp control_read_setup.in ${DATADIR}/$i/control.in
	
	cd ${DATADIR}/$i
	srun --exclusive -n 1 $AIMS < /dev/null > temp.out 
	cd -
	
	cp control_read_predict.in ${DATADIR}/$i/control.in
	python move_data_in.py --predict -i $i
	
	cd ${DATADIR}/$i
	
	mv ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	srun --exclusive -n 1 $AIMS < /dev/null > aims_predict.out && mv rho_rebuilt_ri.out rho_ml.out && mv ri_restart_coeffs.out ri_restart_coeffs_ml.out &

	cd -
done

wait
