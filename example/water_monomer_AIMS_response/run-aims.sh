export OMP_NUM_THREADS=1
ulimit -s unlimited

AIMS=~/aims.x

DATADIR=./data

python -m salted.aims.make_geoms

n=$(ls $DATADIR/geoms | grep -c 'in')
for (( i=1; i<=$n; i++ )); do
	mkdir ${DATADIR}/$i
	cp control.in ${DATADIR}/$i
	cp ${DATADIR}/geoms/$i.in ${DATADIR}/$i/geometry.in
	cd ${DATADIR}/$i

	mpirun -np 1 $AIMS < /dev/null > aims.out && mv  rho_rebuilt_ri.out rho_df.out && mv ri_restart_coeffs.out ri_restart_coeffs_df.out &

	cd -
done

wait

mpirun -np 16 python -m salted.aims.move_data
