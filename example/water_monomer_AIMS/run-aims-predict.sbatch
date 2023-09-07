#! /bin/bash -l

#SBATCH -o ./predict.out.%j
#SBATCH -e ./predict.err.%j
#SBATCH -J predict
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800

mpsd-modules dev-23a
module load gcc/11.3.0 openmpi/4.1.4 netlib-scalapack openblas cmake anaconda3 hdf5 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPSD_GSL_ROOT/lib:$MPSD_OPENBLAS_ROOT/lib:$MPSD_NETLIB_SCALAPACK_ROOT/lib:$MPSD_HDF5_ROOT/lib
eval "$(conda shell.bash hook)"
conda activate mypy3

export OMP_NUM_THREADS=1
ulimit -s unlimited

DATADIR='qmdata/predicted_data'
AIMS=~/aims.master.gnu.x

mkdir $DATADIR

python -m salted.aims.make_geoms --predict

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir $DATADIR/$i
	cp control_read_setup.in ${DATADIR}/$i/control.in
	cp $DATADIR/geoms/$i.in $DATADIR/$i/geometry.in
	
	cd ${DATADIR}/$i
	srun --exclusive -n 1 $AIMS < /dev/null > temp.out 
	cd -
done

wait 

python -m salted.aims.move_data_in 

for (( i=1; i<=$n; i++ )); do
	cp control_read.in ${DATADIR}/$i/control.in
	
	cd ${DATADIR}/$i
	
	mv ri_restart_coeffs_predicted.out ri_restart_coeffs.out
	srun --exclusive -n 1 $AIMS < /dev/null > aims_predict.out && mv rho_rebuilt_ri.out rho_ml.out && mv ri_restart_coeffs.out ri_restart_coeffs_ml.out &

	cd -
done

wait
