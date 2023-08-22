.PHONY: all clean f2py 

all: f2py 

clean:
	cd lib; rm -f *.o *.so
	cd lib; rm -rf build

LIBDIR = lib 
dummy_build_folder := $(shell mkdir -p $(LIBDIR))

F2PYEXE=$(shell which f2py) 

FCOMPILER='gfortran' 
F90FLAGS='-fopenmp' 
F2PYOPT='-O2'
LIBS='-lgomp'
CC=gcc

# WITH INTEL COMPILERS
#FCOMPILER='intelem' 
#F90FLAGS='-qopenmp' 
#F2PYOPT='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'
#LIBS='-liomp5 -lpthread' # omp libraries with ifort

f2py: lib/ovlp2c.so lib/ovlp3c.so lib/ovlp2cXYperiodic.so lib/ovlp3cXYperiodic.so lib/ovlp2cnonperiodic.so lib/ovlp3cnonperiodic.so lib/equicomb.so lib/equicombfield.so 
#lib/gausslegendre.so lib/neighlist_ewald.so lib/nearfield_ewald.so lib/lebedev.so

lib/ovlp2c.so: src/cp2k/ovlp2c.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp2c.f90 -m ovlp2c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2c.*.so ovlp2c.so

lib/ovlp3c.so: src/cp2k/ovlp3c.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp3c.f90 -m ovlp3c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3c.*.so ovlp3c.so

lib/ovlp2cXYperiodic.so: src/cp2k/ovlp2cXYperiodic.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp2cXYperiodic.f90 -m ovlp2cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cXYperiodic.*.so ovlp2cXYperiodic.so

lib/ovlp3cXYperiodic.so: src/cp2k/ovlp3cXYperiodic.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp3cXYperiodic.f90 -m ovlp3cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cXYperiodic.*.so ovlp3cXYperiodic.so

lib/ovlp2cnonperiodic.so: src/cp2k/ovlp2cnonperiodic.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp2cnonperiodic.f90 -m ovlp2cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cnonperiodic.*.so ovlp2cnonperiodic.so

lib/ovlp3cnonperiodic.so: src/cp2k/ovlp3cnonperiodic.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/cp2k/ovlp3cnonperiodic.f90 -m ovlp3cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cnonperiodic.*.so ovlp3cnonperiodic.so

lib/equicomb.so: src/equicomb.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/equicomb.f90 -m equicomb --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicomb.*.so equicomb.so

lib/equicombfield.so: src/equicombfield.f90
	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/equicombfield.f90 -m equicombfield --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombfield.*.so equicombfield.so

#lib/gausslegendre.so: src/gausslegendre.f90
#	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/gausslegendre.f90 -m gausslegendre --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv gausslegendre.*.so gausslegendre.so
#
#lib/neighlist_ewald.so: src/neighlist_ewald.f90
#	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/neighlist_ewald.f90 -m neighlist_ewald --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv neighlist_ewald.*.so neighlist_ewald.so
#
#lib/nearfield_ewald.so: src/nearfield_ewald.f90
#	cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../src/nearfield_ewald.f90 -m nearfield_ewald --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv nearfield_ewald.*.so nearfield_ewald.so
#
#lib/lebedev.so:
#	cd lib; $(CC) -fPIC -shared -o lebedev.so ../src/Lebedev-Laikov.c

