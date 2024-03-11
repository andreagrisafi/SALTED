import os
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import yaml
from ase.io import read

from salted import basis

sys.path.insert(0, './')
import inp


def read_system(filename=inp.filename):

    # read species
    spelist = inp.species

    # read basis
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    llist = []
    nlist = []
    for spe in spelist:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    nnmax = max(nlist)
    llmax = max(llist)

    # read system
    xyzfile = read(filename,":")
    ndata = len(xyzfile)

    # Define system excluding atoms that belong to species not listed in SALTED input
    atomic_symbols = []
    natoms = np.zeros(ndata,int)
    for iconf in range(len(xyzfile)):
        atomic_symbols.append(xyzfile[iconf].get_chemical_symbols())
        natoms_total = len(atomic_symbols[iconf])
        excluded_species = []
        for iat in range(natoms_total):
            spe = atomic_symbols[iconf][iat]
            if spe not in spelist:
                excluded_species.append(spe)
        excluded_species = set(excluded_species)
        for spe in excluded_species:
            atomic_symbols[iconf] = list(filter(lambda a: a != spe, atomic_symbols[iconf]))
        natoms[iconf] = int(len(atomic_symbols[iconf]))

    # Define maximum number of atoms
    natmax = max(natoms)

    return spelist, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax

def get_atom_idx(ndata,natoms,spelist,atomic_symbols):
    # initialize useful arrays
    atom_idx = {}
    natom_dict = {}
    for iconf in range(ndata):
        for spe in spelist:
            atom_idx[(iconf,spe)] = []
            natom_dict[(iconf,spe)] = 0
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            if spe in spelist:
               atom_idx[(iconf,spe)].append(iat)
               natom_dict[(iconf,spe)] += 1

    return atom_idx,natom_dict

def get_conf_range(rank,size,ntest,testrangetot):
    if rank == 0:
        testrange = [[] for _ in range(size)]
        blocksize = int(ntest/float(size))
#       print(ntest,blocksize)
        if type(testrangetot) is not list: testrangetot = testrangetot.tolist()
        for i in range(size):
            if i == (size-1):
                rem = ntest - (i+1)*blocksize
#               print(i,(i+1)*blocksize,rem)
                if rem < 0:
                    testrange[i] = testrangetot[i*blocksize:ntest]
                else:
                    testrange[i] = testrangetot[i*blocksize:(i+1)*blocksize]
                    for j in range(rem):
                        testrange[j].append(testrangetot[(i+1)*blocksize+j])
            else:
                testrange[i] = testrangetot[i*blocksize:(i+1)*blocksize]
#           print(i,len(testrange[i]))
    else:
        testrange = None

    return testrange


def sort_grid_data(data:np.ndarray) -> np.ndarray:
    """Sort real space grid data
    The grid data is 2D array with 4 columns (x,y,z,value).
    Sort the grid data in the order of x, y, z.

    Args:
        data (np.ndarray): grid data, shape (n,4)

    Returns:
        np.ndarray: sorted grid data, shape (n,4)
    """
    assert data.ndim == 2
    assert data.shape[1] == 4
    data = data[np.lexsort((data[:,2], data[:,1], data[:,0]))]  # last key is primary
    return data



class AttrDict:
    """Access dict keys as attributes

    The attr trick only works for nested dicts.
    One just needs to wrap the dict in an AttrDict object.
    """
    def __init__(self, d: dict):
        self._mydict = d

    def __repr__(self):
        return '\n'.join([f"{k}: {repr(v)}" for k, v in self._mydict.items()])

    def __getattr__(self, name):
        assert name in self._mydict.keys(), f"{name} not in {self._mydict.keys()}"
        value = self._mydict[name]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value


class ParseConfig:
    """Input configuration file parser
    """

    def __init__(self, _dev_inp_fpath: Optional[str]=None):
        if _dev_inp_fpath is None:
            self.inp_fpath = os.path.join(os.getcwd(), 'inp.yaml')
        else:
            self.inp_fpath = _dev_inp_fpath
        assert os.path.exists(self.inp_fpath), f"Input file not found: {self.inp_fpath}"

    def parse_input(self):
        with open(self.inp_fpath) as file:
            inp = yaml.load(file, Loader=self.get_loader())
        if inp is None:
            raise ValueError(f"Input file is empty: {self.inp_fpath}")
        inp = self.check_input(inp)
        return AttrDict(inp)

    def check_input(self, inp:Dict):
        """Check keys (required, optional, not allowed), and value types and ranges

        Just check keys and values one by one, silly err haha"""

        """Format: (key, required, default value, value type, value extra check)
        About required:
            - True -> required
            - False -> optional, will fill in default value if not found
            - None -> optional, will NOT fill in default value if not found
        """
        inp_template: Tuple[Tuple[str, Optional[bool], Any, Any, Optional[Callable]]] = (
            # System definition
            ('saltedname', True, None, str, None),  # salted workflow identifier
            ('dfbasis', True, None, str, None),  # density fitting basis
            ('species', True, None, list, lambda x: all(isinstance(i, str) for i in x)),  # list of species in all geometries
            ('qmcode', True, None, str, lambda x: x.lower() in ('aims', 'pyscf', 'cp2k')),  # quantum mechanical code
            ('parallel', False, True, bool, None),  # if use mpi4py
            ('average', False, True, bool, None),  # if bias the GPR by the average of predictions
            ('field', False, False, bool, None),  # if predict the field response
            # Paths
            ('filename', True, None, str, lambda x: os.path.exists(x)),  # path to geometry file
            ('saltedpath', True, None, str, lambda x: os.path.exists(x)),  # path to SALTED outputs
            ('path2qm', True, None, str, lambda x: os.path.exists(x)),  # path to the QM calculation outputs
            ('predname', True, None, str, None),  # SALTED prediction identifier
            ('predict_data', False, "predictoins", str, None),  # path to the prediction data by QM code
            # Rascaline atomic environment parameters
            ('rep1', True, None, str, lambda x: x in ('rho', 'V')),  # descriptor, rho -> SOAP, V -> LODE
            ('rcut1', True, None, float, lambda x: x > 0),  # cutoff radius
            ('sig1', True, None, float, lambda x: x > 0),  # Gaussian width
            ('nrad1', True, None, int, lambda x: x > 0),  # number of radial basis functions
            ('nang1', True, None, int, lambda x: x > 0),  # number of angular basis functions
            ('neighspe1', True, None, list, lambda x: all(isinstance(i, str) for i in x)),  # list of neighbor species
            ('rep2', True, None, str, lambda x: x in ('rho', 'V')),
            ('rcut2', True, None, float, lambda x: x > 0),
            ('sig2', True, None, float, lambda x: x > 0),
            ('nrad2', True, None, int, lambda x: x > 0),
            ('nang2', True, None, int, lambda x: x > 0),
            ('neighspe2', True, None, list, lambda x: all(isinstance(i, str) for i in x)),
            # Feature sparsification parameters
            ('sparsify', False, False, bool, lambda x: isinstance(x, bool)),  # if sparsify features channels
            ('nsamples', False, 100, int, lambda x: x > 0),  # number of samples for sparsifying feature channel
            ('ncut', False, -1, int, lambda x: (x == -1) or (x > 0)),  # number of features to keep
            # ML variables
            ('z', False, 2.0, float, lambda x: x > 0),  # kernel exponent
            ('Menv', True, None, int, lambda x: x > 0),  # number of sparsified atomic environments
            ('Ntrain', True, None, int, lambda x: x > 0),  # size of the training set (rest is validation set)
            ('trainfrac', False, 1.0, float, lambda x: (x > 0) and (x <= 1.0)),  # fraction of the training set used for training
            ('regul', False, 1e-6, float, lambda x: x > 0),  # regularisation parameter
            ('eigcut', False, 1e-10, float, lambda x: x > 0),  # eigenvalues cutoff
            ('gradtol', False, 1e-5, float, lambda x: x > 0),  # min gradient as stopping criterion for CG minimization
            ('restart', False, False, bool, lambda x: isinstance(x, bool)),  # if restart the minimization
            # Special ML variables for direct minimization
            ('blocksize', None, None, int, lambda x: x > 0),  # block size for matrix inversion
            ('trainsel', False, 'random', str, lambda x: x in ('random', 'sequential')),  # if shuffle the training set
        )

        for key, required, val_default, val_type, extra_check_func in inp_template:
            """apply value defaults"""
            if required is None:
                """special case, optional without default value"""
                if key not in inp:
                    continue  # without checking value
            elif required:
                if key not in inp.keys():
                    raise ValueError(f"Required key not found: {key}")
            elif not required:
                if key not in inp.keys():
                    inp[key] = val_default

            """check values"""
            if not isinstance(inp[key], val_type):
                raise ValueError(f"Value type mismatch: {key} -> {val_type}, {key = }")
            if extra_check_func is not None:
                if not extra_check_func(inp[key]):
                    raise ValueError(f"Extra value check failed: {key} -> {inp[key]}")

        return inp

    def get_loader(self) -> yaml.SafeLoader:
        """Add constructors to the yaml.SafeLoader
        For details, see: https://pyyaml.org/wiki/PyYAMLDocumentation
        """
        loader = yaml.SafeLoader

        """for path concatenation, like !join_path [a, b, c] -> a/b/c"""
        def join_path(loader: yaml.SafeLoader, node: yaml.Node) -> str:
            seq = loader.construct_sequence(node)
            return os.path.join(*seq)
        loader.add_constructor('!join_path', join_path)

        """for scientific notation, like 1e-4 -> float(1e-4)"""
        pattern = re.compile(r'^-?[0-9]+(\.[0-9]*)?[eEdD][-+]?[0-9]+$')
        loader.add_implicit_resolver('!float_sci', pattern, list(u'-+0123456789'))
        def float_sci(loader, node):
            value = loader.construct_scalar(node)
            return float(value)
        loader.add_constructor('!float_sci', float_sci)
        return loader
