.PHONY: all clean cython python LODE

# Install the python packages needed to run SOAPFAST

PIPOPTS=--user

all: cython

LODE: gvectors.so phasecomb.so gcontra.so lebedev.so gausslegendre.so nearfield.so nearfield_ewald.so neighlist_ewald.so lode-cython

gvectors.so:
	cd utils/LODE;f2py3 -c --opt='-O2' gvectors.f90 -m gvectors;mv gvectors.*.so gvectors.so

phasecomb.so:
	cd utils/LODE;f2py3 -c --opt='-O2' phasecomb.f90 -m phasecomb --f90flags="-fopenmp" -lgomp;mv phasecomb.*.so phasecomb.so

nearfield.so:
	cd utils/LODE;f2py3 -c --opt='-O2' nearfield.f90 -m nearfield --f90flags="-fopenmp" -lgomp;mv nearfield.*.so nearfield.so

nearfield_ewald.so:
	cd utils/LODE;f2py3 -c --opt='-O2' nearfield_ewald.f90 -m nearfield_ewald --f90flags="-fopenmp" -lgomp;mv nearfield_ewald.*.so nearfield_ewald.so

neighlist_ewald.so:
	cd utils/LODE;f2py3 -c --opt='-O2' neighlist_ewald.f90 -m neighlist_ewald --f90flags="-fopenmp" -lgomp;mv neighlist_ewald.*.so neighlist_ewald.so

gcontra.so:
	cd utils/LODE;f2py3 -c --opt='-O2' gcontra.f90 -m gcontra --f90flags="-fopenmp" -lgomp;mv gcontra.*.so gcontra.so

lebedev.so:
	cd utils/LODE; gcc -fPIC -shared -o lebedev.so Lebedev-Laikov.c

gausslegendre.so:
	cd utils/LODE;f2py3 -c --opt='-O2' gausslegendre.f90 -m gausslegendre;mv gausslegendre.*.so gausslegendre.so

lode-cython:
	cd utils/LODE;python setup_cython.py build_ext --inplace
	@cd utils/LODE;mv fourier_integrals.*.so fourier_integrals.so

install-deps:
	pip install $(PIPOPTS) numpy ase scipy cython sympy

cython: #install-deps
	cd utils;python setup_cython.py build_ext --inplace
	@cd utils;mv initsoap.*.so initsoap.so;mv build_kernel.*.so build_kernel.so

clean:
	cd utils;rm -rf *.so build;cd LODE;rm -rf *.so build
