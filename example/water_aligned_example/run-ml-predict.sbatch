#! /bin/bash -l

#SBATCH -o ./gen_data.out.%j
#SBATCH -e ./gen_data.err.%j
#SBATCH -J gen_data
#SBATCH --partition=p.ada
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
##SBATCH --mem-per-cpu=6000
#SBATCH --time=8:00:00

### THIS IS JUST AN EXAMPLE OF THE WORKFLOW

### SET UP ENVIRONMENT VARIABLES: (uncomment and edit as needed)

ulimit -s unlimited
ulimit -c 0

#export OMP_NUM_THREADS=1

module load anaconda/3/2023.03  intel/21.7.1   impi/2021.7   mkl/2023.1   mpi4py/3.1.4

### RUN YOUR CODE:

source ~/codes/TENSOAP/env.sh
source ~/codes/SALTED-SALTER/env.sh
export PYTHONPATH=$PWD:$PYTHONPATH

# note that this is just an example. In the command below: remove -p if not periodic, remove -nc 2000 if you do not want 
# to sparsify the descriptors and remove -vf 0.2 is the volumetric scaling does not apply to your system
#
# Note that you will need to run this to multiple times to predict the response to a field along the x, y and z axis
python3 ~/codes/SALTED-SALTER/src/get_averages.py

python3 -u ~/codes/SALTED-SALTER/src/run-tensoap.py -p -nc 2000 -d 1 -vf 0.2 --parallel=16 --predict &> outerrsxp
wait
python3 -u ~/codes/SALTED-SALTER/src/parallel_rkhs_prep.py --predict &> outprkhsx
wait
srun -n 16 python3 -u ~/codes/SALTED-SALTER/src/rkhs-prediction.py  &> outrkhsxp
wait

srun -n 16 python3 -u ~/codes/SALTED-SALTER/src/prediction.py &> outpredictx
wait
