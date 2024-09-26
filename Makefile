.PHONY: all clean f2py init

all: f2py init

clean:
	cd salted; rm -rf lib

LIBDIR = salted/lib
dummy_build_folder := $(shell mkdir -p $(LIBDIR))

F2PYEXE := $(shell which f2py3 || which f2py)  # simple expansion, not recursive, won't be re-evaluated

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

init:
	cd salted/lib; touch __init__.py

f2py: salted/lib/ovlp2c.so salted/lib/ovlp3c.so salted/lib/ovlp2cXYperiodic.so salted/lib/ovlp3cXYperiodic.so salted/lib/ovlp2cnonperiodic.so salted/lib/ovlp3cnonperiodic.so salted/lib/equicomb.so salted/lib/equicombfield.so salted/lib/equicombsparse.so salted/lib/antiequicomb.so salted/lib/antiequicombsparse.so salted/lib/equicombnonorm.so salted/lib/antiequicombnonorm.so salted/lib/kernelequicomb.so salted/lib/equicombfps.so

#salted/lib/gausslegendre.so salted/lib/neighlist_ewald.so salted/lib/nearfield_ewald.so salted/lib/lebedev.so

salted/lib/ovlp2c.so: src/ovlp2c.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp2c.f90 -m ovlp2c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2c.*.so ovlp2c.so

salted/lib/ovlp3c.so: src/ovlp3c.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp3c.f90 -m ovlp3c --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3c.*.so ovlp3c.so

salted/lib/ovlp2cXYperiodic.so: src/ovlp2cXYperiodic.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp2cXYperiodic.f90 -m ovlp2cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cXYperiodic.*.so ovlp2cXYperiodic.so

salted/lib/ovlp3cXYperiodic.so: src/ovlp3cXYperiodic.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp3cXYperiodic.f90 -m ovlp3cXYperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cXYperiodic.*.so ovlp3cXYperiodic.so

salted/lib/ovlp2cnonperiodic.so: src/ovlp2cnonperiodic.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp2cnonperiodic.f90 -m ovlp2cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp2cnonperiodic.*.so ovlp2cnonperiodic.so

salted/lib/ovlp3cnonperiodic.so: src/ovlp3cnonperiodic.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/ovlp3cnonperiodic.f90 -m ovlp3cnonperiodic --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv ovlp3cnonperiodic.*.so ovlp3cnonperiodic.so

salted/lib/equicomb.so: src/equicomb.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/equicomb.f90 -m equicomb --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicomb.*.so equicomb.so

salted/lib/equicombfield.so: src/equicombfield.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/equicombfield.f90 -m equicombfield --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombfield.*.so equicombfield.so

salted/lib/equicombsparse.so: src/equicombsparse.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/equicombsparse.f90 -m equicombsparse --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombsparse.*.so equicombsparse.so

salted/lib/antiequicomb.so: src/antiequicomb.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/antiequicomb.f90 -m antiequicomb --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv antiequicomb.*.so antiequicomb.so

salted/lib/antiequicombsparse.so: src/antiequicombsparse.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/antiequicombsparse.f90 -m antiequicombsparse --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv antiequicombsparse.*.so antiequicombsparse.so

salted/lib/equicombnonorm.so: src/equicombnonorm.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/equicombnonorm.f90 -m equicombnonorm --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombnonorm.*.so equicombnonorm.so

salted/lib/antiequicombnonorm.so: src/antiequicombnonorm.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/antiequicombnonorm.f90 -m antiequicombnonorm --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv antiequicombnonorm.*.so antiequicombnonorm.so

salted/lib/kernelequicomb.so: src/kernelequicomb.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/kernelequicomb.f90 -m kernelequicomb --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv kernelequicomb.*.so kernelequicomb.so

salted/lib/equicombfps.so: src/equicombfps.f90
	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/equicombfps.f90 -m equicombfps --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv equicombfps.*.so equicombfps.so

#salted/lib/gausslegendre.so: src/gausslegendre.f90
#	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/gausslegendre.f90 -m gausslegendre --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv gausslegendre.*.so gausslegendre.so
#
#salted/lib/neighlist_ewald.so: src/neighlist_ewald.f90
#	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/neighlist_ewald.f90 -m neighlist_ewald --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv neighlist_ewald.*.so neighlist_ewald.so
#
#salted/lib/nearfield_ewald.so: src/nearfield_ewald.f90
#	cd salted/lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../../src/nearfield_ewald.f90 -m nearfield_ewald --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS); mv nearfield_ewald.*.so nearfield_ewald.so
#
#salted/lib/lebedev.so:
#	cd salted/lib; $(CC) -fPIC -shared -o lebedev.so ../../src/Lebedev-Laikov.c

