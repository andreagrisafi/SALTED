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

module use /opt/easybuild/modules/all
module load Anaconda3

source activate mypy3

if [[ 1 -eq 2 ]]; then
for i in 0 1 2 3 4 5 6 7 8; do
   ~/TENSOAP/soapfast/get_power_spectrum.py -p -f water_monomers_1k_periodic.xyz -lm ${i} -c H O -s H O -o mldata/soaps/FEAT-${i} 
done
fi

if [[ 1 -eq 1 ]]; then
srun -n 10 python move_data.py
python ~/SALTED/src/get_averages.py
python ~/SALTED/src/rkhs.py
srun -n 10 python ~/SALTED/src/feature_vector.py
fi

if [[ 1 -eq 1 ]]; then
for i in {6..6}; do
	sed -i "s/regul.*/regul = 1e-$i/" inp.py
	srun -n 1 python ~/SALTED/src/minimize_loss.py > reg_$i.out
	python ~/SALTED/src/validation.py >> reg_$i.out
done
fi

if [[ 1 -eq 2 ]]; then
for i in {1..4}; do
	j=$(bc<<<$i*0.25)
	sed -i "s/trainfrac.*/trainfrac = $j/" inp.py
	srun -n 10 python ~/SALTED-private-modified/src/minimize_loss-parallel.py > frac_$i.out
	python ~/SALTED-private-modified/src/validation.py >> frac_$i.out
done
fi

