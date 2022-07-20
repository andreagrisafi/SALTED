#! /bin/bash -l

#SBATCH -o ./gen_data.out.%j
#SBATCH -e ./gen_data.err.%j
#SBATCH -J gen_data
#SBATCH --partition=mpsd
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800

export OMP_NUM_THREADS=1
ulimit -s unlimited
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/compilers_and_libraries/linux/lib/intel64/:$MKL_HOME/lib/intel64/:$HOME/.local/lib

QMDIR='qmdata1/'
AIMS=~/aims.master.mkl.x

DATADIR=${QMDIR}data

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir ${DATADIR}/$i
	cp control.in ${DATADIR}/$i
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	cd ${DATADIR}/$i

	srun -n 1 $AIMS < /dev/null > aims.out && mv  rho_rebuilt_ri.out rho_df.out && mv ri_restart_coeffs.out ri_restart_coeffs_df.out &

	cd -
done

wait
