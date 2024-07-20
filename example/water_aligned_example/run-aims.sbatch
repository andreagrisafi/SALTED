#! /bin/bash -l

#SBATCH -o ./gen_data.out.%j
#SBATCH -e ./gen_data.err.%j
#SBATCH -J gen_data
#SBATCH --partition=p.ada
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --mem-per-cpu=6000
#SBATCH --time=1:00:00


export OMP_NUM_THREADS=1
ulimit -s unlimited
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$INTEL_HOME/compilers_and_libraries/linux/lib/intel64/:$MKL_HOME/lib/intel64/:$HOME/.local/lib

QMDIR='/ada/ptmp/mpsd/alewis/water_aligned/qmdata/'
AIMS=~/aims.x

DATADIR=${QMDIR}data

python make_geoms.py

n=$(ls $DATADIR/geoms | grep -c 'in')
for (( i=1; i<=$n; i++ )); do
	let j=$i-1
	mkdir ${DATADIR}/$i
	cp control.in ${DATADIR}/$i
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	cd ${DATADIR}/$i

	srun --exclusive -n 16 $AIMS < /dev/null > aims.out && mv  rho_rebuilt_ri.out rho_df.out && mv ri_restart_coeffs.out ri_restart_coeffs_df.out &

	cd -
done

wait

srun -n 72 python move_data.py
srun -n 72 python move_rho1_data.py
