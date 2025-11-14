.PHONY: all clean f2py init

all: f2py init

clean:
	cd salted; rm -rf lib

LIBDIR := salted/lib
dummy_build_folder := $(shell mkdir -p $(LIBDIR))

# Python: availability and version
PYTHON_CHECK := $(shell python -c "import sys; sys.exit(0)" 2>/dev/null && echo "ok" || echo "fail")
ifeq ($(PYTHON_CHECK),fail)
    $(error Python is not installed or not in PATH. Please install it.)
endif
PYTHON_VERSION := $(shell python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# NumPy: availability and version
NUMPY_CHECK := $(shell python -c "import numpy; import sys; sys.exit(0)" 2>/dev/null && echo "ok" || echo "fail")
ifeq ($(NUMPY_CHECK),fail)
    $(error NumPy is not installed. Please install it.)
endif
NUMPY_VERSION := $(shell python -c "import numpy; print(numpy.__version__)")

# packaging: availability and version
PACKAGING_CHECK := $(shell python -c "import packaging; import sys; sys.exit(0)" 2>/dev/null && echo "ok" || echo "fail")
ifeq ($(PACKAGING_CHECK),fail)
    $(error packaging module is not installed. Please install it.)
endif
PACKAGING_VERSION := $(shell python -c "import packaging; print(packaging.__version__)")

# Determine if we should use meson backend (NumPy >= 1.26.0) or distutils backend
# https://numpy.org/doc/stable/reference/distutils_status_migration.html
# find more with: conda run -n salted_tut python -m numpy.f2py -c --help-fcompiler
# this command says: distutils has been deprecated since NumPy 1.26.x
USE_MESON := $(shell python -c "from packaging import version; import numpy; print('yes' if version.parse(numpy.__version__) >= version.parse('1.26.0') else 'no')" 2>/dev/null || echo "no")

# Check module version, set backend flag and compiler configuration
ifeq ($(USE_MESON),yes)
    # meson: availability and version (required for meson backend)
    MESON_CHECK := $(shell which meson >/dev/null 2>&1 && echo "ok" || echo "fail")
    ifeq ($(MESON_CHECK),fail)
        $(error meson is not installed. Required for NumPy >= 1.26 (meson backend). Please install it.)
    endif
    MESON_VERSION := $(shell meson --version)
    # Meson backend: use FC environment variable, not --fcompiler
    # Meson uses environment variables for compiler flags, NOT --f90flags (that's distutils-only)
    F2PY_COMPILER_VARS := FC='gfortran' FFLAGS='-fopenmp -O2'
    BACKEND_FLAG := --backend meson
    F2PY_COMPILER_FLAGS :=
    F2PY_LIBS := -lgomp
    $(info [SALTED Build] Using meson backend (Python $(PYTHON_VERSION), NumPy $(NUMPY_VERSION), Meson $(MESON_VERSION)))
else
    # setuptools: availability and version (required for distutils backend)
    SETUPTOOLS_CHECK := $(shell python -c "import setuptools; import sys; sys.exit(0)" 2>/dev/null && echo "ok" || echo "fail")
    ifeq ($(SETUPTOOLS_CHECK),fail)
        $(error setuptools is not installed. Required for NumPy < 1.26 (distutils backend). Please install it.)
    endif
    SETUPTOOLS_VERSION := $(shell python -c "import setuptools; print(setuptools.__version__)")
    SETUPTOOLS_OK := $(shell python -c "from packaging import version; import setuptools; print('yes' if version.parse(setuptools.__version__) < version.parse('60.0') else 'no')")
    # Distutils backend: use --fcompiler and --f90flags
    ifeq ($(SETUPTOOLS_OK),no)
        $(error setuptools >= 60.0 detected ($(SETUPTOOLS_VERSION)). Consider downgrading to setuptools<60.0 for distutils backend compatibility, e.g. pip install setuptools==59.8.0)
    endif
    F2PY_COMPILER_VARS :=
    BACKEND_FLAG :=
    F2PY_COMPILER_FLAGS := --fcompiler='gnu95' --f90flags='-fopenmp -O2'
    F2PY_LIBS := -lgomp
    $(info [SALTED Build] Using distutils backend (Python $(PYTHON_VERSION), NumPy $(NUMPY_VERSION), setuptools $(SETUPTOOLS_VERSION)))
endif

# WITH INTEL COMPILERS
#FCOMPILER := 'intelem'
#F90FLAGS := '-qopenmp'
#F2PYOPT := "--opt='-O3 -unroll-aggressive -qopt-prefetch -qopt-reportr5'"
#LIBS := '-liomp5 -lpthread' # omp libraries with ifort

init:
	cd salted/lib; touch __init__.py

f2py: salted/lib/ovlp2c.so salted/lib/ovlp3c.so salted/lib/ovlp2cXYperiodic.so salted/lib/ovlp3cXYperiodic.so salted/lib/ovlp2cnonperiodic.so salted/lib/ovlp3cnonperiodic.so salted/lib/equicomb.so salted/lib/equicombfield.so salted/lib/equicombsparse.so salted/lib/antiequicomb.so salted/lib/antiequicombsparse.so salted/lib/equicombnonorm.so salted/lib/antiequicombnonorm.so salted/lib/kernelequicomb.so salted/lib/kernelnorm.so salted/lib/equicombfps.so

# Pattern rule for compiling all Fortran modules
# The % wildcard matches the module name, and $* expands to the matched part
salted/lib/%.so: src/%.f90
	cd salted/lib; $(F2PY_COMPILER_VARS) python -m numpy.f2py $(F2PYOPT) -c ../../src/$*.f90 -m $* $(BACKEND_FLAG) $(F2PY_COMPILER_FLAGS) $(F2PY_LIBS); mv $*.*.so $*.so
