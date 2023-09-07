from setuptools import setup

setup(
    name='salted',
    version='3.0.0',    
    description='Symmetry-Adapted Learning of Three-Dimensional Electron Densities',
    url='https://github.com/andreagrisafi/SALTED',
    author='Andrea Grisafi, Alan Lewis',
    author_email='andrea.grisafi@ens.psl.eu, alan.m.lewis@york.ac.uk',
    license='GNU GENERAL PUBLIC LICENSE',
    packages=['salted','salted.cp2k','salted.pyscf','salted.aims','salted.lib'],
    install_requires=['mpi4py','rascaline','ase','numpy','scipy','h5py','sympy'],
    include_package_data=True,
    package_data={"salted": ["salted/lib/*.so"]},
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
    ],
)
