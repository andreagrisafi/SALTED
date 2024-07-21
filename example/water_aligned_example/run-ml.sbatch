#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ML.out
##SBATCH -e ./ML-setup.err
# Initial working directory:
##SBATCH -D ./
# Job Name:
#SBATCH -J ML
# Queue (Partition):
#SBATCH --partition=nodes
#SBATCH --nodes=1
#SBATCH --ntasks=50
# for OpenMP:
##SBATCH --cpus-per-task=1
#
# Memory usage of the job [MB], 3800 MB per task:
#SBATCH --mem-per-cpu=3800
#
#SBATCH --mail-type=none
#
# Wall clock limit:
#SBATCH --time=01:00:00

### SET UP ENVIRONMENT VARIABLES: (uncomment and edit as needed)

ulimit -s unlimited
ulimit -c 0

export OMP_NUM_THREADS=1

### RUN YOUR CODE:

source ~/TENSOAP/env.sh
source ~/SALTED/env.sh
export PYTHONPATH=$PWD

python3 ~/SALTED/src/run-tensoap.py -p -nc 0 -d 1 --parallel=50
python3 ~/SALTED/src/parallel_rkhs_prep.py
srun -n 50 python3 ~/SALTED/src/rkhs.py
srun -n 50 python3 ~/SALTED/src/feature_vector.py

python3 ~/SALTED/src/get_averages.py
for i in {8..8}; do
        sed -i "s/regul.*/regul = 1e-$i/" inp.py
        srun -n 50 python3 ~/SALTED/src/minimize_loss.py
        srun -n 50 python3 ~/SALTED/src/validation.py > val_x.out
done

