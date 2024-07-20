#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ML.out
##SBATCH -e ./ML-setup.err
# Initial working directory:
##SBATCH -D ./
# Job Name:
#SBATCH -J ML
# Queue (Partition):
#SBATCH --partition=public
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
# for OpenMP:
##SBATCH --cpus-per-task=1
#
# Memory usage of the job [MB], 3800 MB per task:
#SBATCH --mem-per-cpu=3800
#
#SBATCH --mail-type=none
#
# Wall clock limit:
#SBATCH --time=00:10:00

### SET UP ENVIRONMENT VARIABLES: (uncomment and edit as needed)

ulimit -s unlimited
ulimit -c 0

export OMP_NUM_THREADS=1

### RUN YOUR CODE:

#module purge
mpsd-modules dev-23a
module load gcc/11.3.0 openmpi/4.1.4 netlib-scalapack openblas cmake anaconda3 
eval "$(conda shell.bash hook)"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPSD_GSL_ROOT/lib:$MPSD_OPENBLAS_ROOT/lib:$MPSD_NETLIB_SCALAPACK_ROOT/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPSD_GSL_ROOT/lib:$MPSD_OPENBLAS_ROOT/lib:$MPSD_NETLIB_SCALAPACK_ROOT/lib:$MPSD_PY_MPI4PY_ROOT/lib:$MPSD_OPENMPI_ROOT/lib

#echo $LD_LIBRARY_PATH
#module avail

conda deactivate
conda activate mpi4py
module load py-mpi4py

source ~/TENSOAP/env.sh
source ~/SALTED/env.sh
export PYTHONPATH=$PWD

if [[ 1 -eq 2 ]]; then
srun -n 25 python move_data.py
python ~/SALTED/src/get_averages.py
fi

if [[ 1 -eq 2 ]]; then
#python $SALTEDPATH/run-tensoap.py -p -nc 0 -d 1
#python ~/SALTED/src/rkhs.py
python ~/SALTED/src/feature_vector.py
fi

if [[ 1 -eq 1 ]]; then
#python ~/SALTED/src/get_averages.py
for i in {8..8}; do
	sed -i "s/regul.*/regul = 1e-$i/" inp.py
#	python ~/SALTED/src/minimize_loss.py
	python ~/SALTED/src/validation.py > val_x_2.out
done
fi

if [[ 1 -eq 2 ]]; then
python $SALTEDPATH/run-tensoap.py -p -nc 0 --vol_frac 0.35 -rc 5.0 --predict
python ~/SALTED/src/rkhs-prediction.py
python ~/SALTED/src/prediction.py
fi

if [[ 1 -eq 2 ]]; then
for i in {1..4}; do
	j=$(bc<<<$i*0.25)
	sed -i "s/trainfrac.*/trainfrac = $j/" inp.py
	srun -n 10 python ~/SALTED-private-modified/src/minimize_loss-parallel.py > frac_$i.out
	python ~/SALTED-private-modified/src/validation.py >> frac_$i.out
done
fi

