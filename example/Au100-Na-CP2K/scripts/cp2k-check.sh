#!/bin/bash
#PBS -q beta
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -N test
#PBS -l walltime=03:00:00
#PBS -j oe

module load gcc/11.2 python/3.9 openMPI/4.1.2-gcc112 

export INTRISICS=1 # fix for xsmm vector instructions
export MPI_HOME=/opt/dev/libs/OpenMPI-4.1.2-gcc112

#export omp_num_threads=2


cd /scratchbeta/grisafia/Au-fcc100-223-Na/runs/

for k in {33..33}
do
echo '# Predicted RI coefficients' > /scratchbeta/grisafia/Au-fcc100-223-Na/runs/conf_${k}/PREDICTED_COEFFS.dat; cat /scratchbeta/grisafia/Au-fcc100-223-Na/predictions-nofield/M50_eigcut-10/N32_reg-8/COEFFS-${k}.dat >> /scratchbeta/grisafia/Au-fcc100-223-Na/runs/conf_${k}/PREDICTED_COEFFS.dat 
cd conf_${k}
mpirun /home/grisafia/source/cp2k/exe/local-mpicc/cp2k.popt -i gpw-cubes-RI.inp -o out-cubes-RI.cp2k
cd ../
done
