from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='salted',
    version='3.0.0',    
    description='Symmetry-Adapted Learning of Three-Dimensional Electron Densities',
    url='https://github.com/andreagrisafi/SALTED',
    author='Andrea Grisafi, Alan Lewis',
    author_email='andrea.grisafi@ens.psl.eu, alan.m.lewis@york.ac.uk',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=['salted','salted.cp2k','salted.pyscf','salted.aims','salted.lib', "salted.cython"],
    install_requires=['rascaline','ase','numpy','scipy','sympy', "tqdm", "cython"],
    include_package_data=True,
    package_data={"salted": ["salted/lib/*.so"]},
    ext_modules = cythonize("cython/dm2df_fast_reorder.pyx"),
    include_dirs=[numpy.get_include()],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
