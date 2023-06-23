.PHONY: all clean f2py 

F2PYEXE=$(shell which f2py) 

FCOMPILER='gfortran' 
F90FLAGS='-fopenmp' 
F2PYOPT='-O2'
LIBS='-lgomp'

# WITH INTEL COMPILERS
#FCOMPILER='intelem' 
#F90FLAGS='-qopenmp' 
#F2PYOPT='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'
#LIBS='-liomp5 -lpthread' # omp libraries with ifort

f2py: src/lib/ovlp2c.so src/lib/ovlp3c.so src/lib/ovlp2cXYperiodic.so src/lib/ovlp3cXYperiodic.so src/lib/ovlp2cnonperiodic.so src/lib/ovlp3cnonperiodic.so src/lib/pouter.so src/lib/equicomb.so src/lib/equicombfield.so

src/lib/ovlp2c.so: src/ovlp2c.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp2c.f90 -m ovlp2c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2c.*.so ovlp2c.so

src/lib/ovlp3c.so: src/ovlp3c.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp3c.f90 -m ovlp3c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3c.*.so ovlp3c.so

src/lib/ovlp2cXYperiodic.so: src/ovlp2cXYperiodic.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp2cXYperiodic.f90 -m ovlp2cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cXYperiodic.*.so ovlp2cXYperiodic.so

src/lib/ovlp3cXYperiodic.so: src/ovlp3cXYperiodic.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp3cXYperiodic.f90 -m ovlp3cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cXYperiodic.*.so ovlp3cXYperiodic.so

src/lib/ovlp2cnonperiodic.so: src/ovlp2cnonperiodic.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp2cnonperiodic.f90 -m ovlp2cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cnonperiodic.*.so ovlp2cnonperiodic.so

src/lib/ovlp3cnonperiodic.so: src/ovlp3cnonperiodic.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../ovlp3cnonperiodic.f90 -m ovlp3cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cnonperiodic.*.so ovlp3cnonperiodic.so

src/lib/pouter.so: src/pouter.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../pouter.f90 -m pouter --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv pouter.*.so pouter.so

src/lib/equicomb.so: src/equicomb.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../equicomb.f90 -m equicomb --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicomb.*.so equicomb.so

src/lib/equicombfield.so: src/equicombfield.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../equicombfield.f90 -m equicombfield --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombfield.*.so equicombfield.so

all: f2py 

clean:
	cd src; cd lib; rm -f *.o *.so
	cd src; cd lib; rm -rf build
all: f2py 

clean:
	cd src; cd lib; rm -f *.o *.so
	cd src; cd lib; rm -rf build
