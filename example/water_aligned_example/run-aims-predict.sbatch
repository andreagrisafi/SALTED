#! /bin/bash -l

#SBATCH -o ./pred_data.out.%j
#SBATCH -e ./pred_data.err.%j
#SBATCH -J pred_data
#SBATCH --partition=p.ada
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --mem-per-cpu=6000
#SBATCH --time=1:00:00


export OMP_NUM_THREADS=1
ulimit -s unlimited
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/compilers_and_libraries/linux/lib/intel64/:$MKL_HOME/lib/intel64/:$HOME/.local/lib

QMDIR='/ada/ptmp/mpsd/alewis/water_aligned2/qmdata/'
AIMS=~/aims.x

DATADIR=${QMDIR}predict_data

python make_geoms.py --predict

n=$(ls $DATADIR/geoms | grep -c 'in')
for (( i=1; i<=$n; i++ )); do
	let j=$i-1
	mkdir ${DATADIR}/$i
	cp control_read.in ${DATADIR}/$i/control.in
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	
	python move_data_in.py -i $i -r --predict
	cd ${DATADIR}/$i

	cp ri_rho1_restart_coeffs_predicted_1.out ri_rho1_restart_coeffs_1.out
	cp ri_rho1_restart_coeffs_predicted_2.out ri_rho1_restart_coeffs_2.out
	cp ri_rho1_restart_coeffs_predicted_3.out ri_rho1_restart_coeffs_3.out

	srun --exclusive -n 18 -c 4 $AIMS < /dev/null > aims_predict.out &
	cd -

done

