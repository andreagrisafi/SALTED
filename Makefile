.PHONY: all clean f2py

F2PYEXE=$(shell which f2py2.7) 

FCOMPILER='gfortran' 
F90FLAGS='-fopenmp' 
F2PYOPT='-O3'
LIBS='-lgomp'

# WITH INTEL COMPILERS
#FCOMPILER='intelem' 
#F90FLAGS='-qopenmp' 
#F2PYOPT='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'
#LIBS='-liomp5 -lpthread' # omp libraries with ifort


f2py: src/lib/rmatrix.so src/lib/matrices.so src/lib/prediction.so 

src/lib/rmatrix.so: src/rmatrix.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../rmatrix.f90 -m rmatrix --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS) 

src/lib/matrices.so: src/matrices.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../matrices.f90 -m matrices --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS)  $(LIBS)

src/lib/prediction.so: src/prediction.f90
	cd src; cd lib; $(F2PYEXE) -c --opt=$(F2PYOPT) ../prediction.f90 -m prediction --fcompiler=$(FCOMPILER) --f90flags=$(F90FLAGS) $(LIBS)

all: f2py 

clean:
	cd src; cd lib; rm -f *.o *.so
	cd src; cd lib; rm -rf build
