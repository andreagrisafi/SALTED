#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./ML-water.out
##SBATCH -e ./ML-setup.err
# Initial working directory:
##SBATCH -D ./
# Job Name:
#SBATCH -J ML-water
# Queue (Partition):
#SBATCH --partition=XXXX
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
#SBATCH --time=8:00:00

### SET UP ENVIRONMENT VARIABLES: (uncomment and edit as needed)


### RUN YOUR CODE:

python -m salted.get_basis_info
python -m salted.initialize
python -m salted.sparse_selection
srun -n 16 python -m salted.sparse_descriptor

python -m salted.rkhs_projector
srun -n 16 python -m salted.rkhs_vector

srun -n 16 python -m salted.hessian_matrix
python -m salted.solve_regression
srun -n 16 python -m salted.validation > validation.out

