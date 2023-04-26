#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ML-water.out
##SBATCH -e ./ML-setup.err
# Initial working directory:
##SBATCH -D ./
# Job Name:
#SBATCH -J ML-water
# Queue (Partition):
#SBATCH --partition=mpsd
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=4
##SBATCH --ntasks=32
# for OpenMP:
##SBATCH --cpus-per-task=1
#
# Memory usage of the job [MB], 3800 MB per task:
#SBATCH --mem-per-cpu=3800
#
#SBATCH --mail-type=none
#
# Wall clock limit:
##SBATCH --time=3:30:00
#SBATCH --time=8:00:00

### SET UP ENVIRONMENT VARIABLES: (uncomment and edit as needed)

ulimit -s unlimited
ulimit -c 0

export OMP_NUM_THREADS=32

### RUN YOUR CODE:

source ~/SALTED/env.sh
source ~/TENSOAP/env.sh


python $SALTEDPATH/get_averages.py

python $SALTEDPATH/run-tensoap.py -p -nc 0 --parallel 32

srun -n 32 python $SALTEDPATH/rkhs.py
srun -n 32 python $SALTEDPATH/feature_vector.py

srun -n 32 python $SALTEDPATH/minimize_loss.py 
srun -n 32 python $SALTEDPATH/validation.py > validation.out

