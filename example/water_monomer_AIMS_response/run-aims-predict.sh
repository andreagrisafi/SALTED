export OMP_NUM_THREADS=1
ulimit -s unlimited

AIMS=~/aims.x

DATADIR=./aims_pred_data

python -m salted.aims.make_geoms --predict

n=$(ls $DATADIR/geoms | grep -c 'in')

for (( i=1; i<=$n; i++ )); do
	mkdir $DATADIR/$i
	cp control_read.in ${DATADIR}/$i/control.in
	cp $DATADIR/geoms/$i.in $DATADIR/$i/geometry.in
done

wait 

python -m salted.aims.move_data_in 

for (( i=1; i<=$n; i++ )); do
	cd ${DATADIR}/$i
        mv ri_rho1_restart_coeffs_predicted_1.out ri_rho1_restart_coeffs_1.out
        mv ri_rho1_restart_coeffs_predicted_2.out ri_rho1_restart_coeffs_2.out
        mv ri_rho1_restart_coeffs_predicted_3.out ri_rho1_restart_coeffs_3.out

	mpirun -np 1 $AIMS < /dev/null > aims_predict.out &

	cd -
done

wait
