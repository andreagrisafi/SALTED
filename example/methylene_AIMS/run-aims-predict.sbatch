#! /bin/bash -l

#SBATCH -o ./predict.out.%j
#SBATCH -e ./predict.err.%j
#SBATCH -J predict
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800
#SBATCH --time=00:05:00

#module purge
#module load intel Python/3.12.3-GCCcore-13.3.0 HDF5/1.14.5-iimpi-2024a
#module load SciPy-bundle/2024.05-gfbf-2024a
export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi2.so


export OMP_NUM_THREADS=1
ulimit -s unlimited

DATADIR='qmdata/aims_pred_data'
AIMS='aims.x'

mkdir $DATADIR

python -m salted.aims.make_geoms --predict

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir $DATADIR/$i
	cp control_read.in ${DATADIR}/$i/control.in
	cp $DATADIR/geoms/$i.in $DATADIR/$i/geometry.in
done

wait 

mpirun -np 16 python -m salted.aims.move_data_in 

for (( i=1; i<=$n; i++ )); do
	cd ${DATADIR}/$i
	mv ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	mv ri_restart_coeffs_predicted_beta.out ri_restart_coeffs_beta.out
	mpirun -np 1 $AIMS < /dev/null > aims_predict.out && mv rho_rebuilt_ri.out rho_ml.out && mv ri_restart_coeffs.out ri_restart_coeffs_ml.out && mv ri_restart_coeffs_beta.out ri_restart_coeffs_ml_beta.out &

	cd -
done

wait
