#! /bin/bash -l

#SBATCH -o ./gen_data.out.%j
#SBATCH -e ./gen_data.err.%j
#SBATCH -J gen_data
#SBATCH --partition=XXXX
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=3800
#SBATCH --cpus-per-task=1

export OMP_NUM_THREADS=1
ulimit -s unlimited

QMDIR='qmdata/'
AIMS=~/aims.master.mkl.x

DATADIR=${QMDIR}data

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir ${DATADIR}/$i
	cp control.in ${DATADIR}/$i
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	cd ${DATADIR}/$i

	srun --exclusive -n 1 $AIMS < /dev/null > aims.out && mv  rho_rebuilt_ri.out rho_df.out && mv ri_restart_coeffs.out ri_restart_coeffs_df.out &

	cd -
done

wait
