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

def get_feats_projs(species,lmax):
    import h5py
    import os.path as osp
    Vmat = {}
    Mspe = {}
    power_env_sparse = {}
    sdir = osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}")
    features = h5py.File(osp.join(sdir,f"FEAT_M-{inp.Menv}.h5"),'r')
    projectors = h5py.File(osp.join(sdir,f"projector_M{inp.Menv}_zeta{inp.z}.h5"),'r')
    for spe in species:
        for lam in range(lmax[spe]+1):
             # load RKHS projectors
             Vmat[(lam,spe)] = projectors["projectors"][spe][str(lam)][:]
             # load sparse equivariant descriptors
             power_env_sparse[(lam,spe)] = features["sparse_descriptors"][spe][str(lam)][:]
             if lam == 0:
                 Mspe[spe] = power_env_sparse[(lam,spe)].shape[0]
             # precompute projection on RKHS if linear model
             if inp.z==1:
                 power_env_sparse[(lam,spe)] = np.dot(
                     Vmat[(lam,spe)].T, power_env_sparse[(lam,spe)]
                 )
    features.close()
    projectors.close()

    return Vmat,Mspe,power_env_sparse

class AttrDict:
    """Access dict keys as attributes

    The attr trick only works for nested dicts.
    One just needs to wrap the dict in an AttrDict object.
    """
    def __init__(self, d: dict):
        self._mydict = d

    def __repr__(self):
        def rec_repr(d: dict, offset=''):
            return '\n'.join([
                offset + f"{k}: {repr(v)}"
                if not isinstance(v, dict)
                else offset + f"{k}:\n{rec_repr(v, offset+'  ')}"
                for k, v in d.items()
            ])
        return rec_repr(self._mydict, offset='')

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
        """Initialize configuration parser

        Args:
            _dev_inp_fpath (Optional[str], optional): Path to the input file. Defaults to None.
                Don't use this argument, it's for testing only!!!
        """
        if _dev_inp_fpath is None:
            self.inp_fpath = os.path.join(os.getcwd(), 'inp.yaml')
        else:
            self.inp_fpath = _dev_inp_fpath
        assert os.path.exists(self.inp_fpath), f"Input file not found: {self.inp_fpath}"

    def parse_input(self) -> AttrDict:
        """Parse input file
        Procedure:
        - get loader (for constructors and resolvers)
        - load yaml

        Returns:
            AttrDict: Parsed input file
        """
        with open(self.inp_fpath) as file:
            inp = yaml.load(file, Loader=self.get_loader())
        if inp is None:
            raise ValueError(f"Input file is empty: {self.inp_fpath}")
        inp = self.check_input(inp)
        return AttrDict(inp)

    def get_all_params(self) -> Tuple:
        """return all parameters with a tuple

        Please copy & paste:
        ```python
        (saltedname, saltedpath,
         filename, species, average, field, parallel,
         path2qm, qmcode, qmbasis, dfbasis,
         filename_pred, predname, predict_data,
         rep1, rcut1, sig1, nrad1, nang1, neighspe1,
         rep2, rcut2, sig2, nrad2, nang2, neighspe2,
         sparsify, nsamples, ncut,
         z, Menv, Ntrain, trainfrac, regul, eigcut, gradtol, restart,
         blocksize, trainsel) = ParseConfig().get_all_params()
        ```
        """
        inp = self.parse_input()
        sparsify = False if inp.dcpt.sparsify.ncut == 0 else True
        return (
            inp.salted.saltedname, inp.salted.saltedpath,
            inp.sys.filename, inp.sys.species, inp.sys.average, inp.sys.field, inp.sys.parallel,
            inp.qm.path2qm, inp.qm.qmcode, inp.qm.qmbasis, inp.qm.dfbasis,
            inp.pred.filename_pred, inp.pred.predname, inp.pred.predict_data,
            inp.dcpt.rep1, inp.dcpt.rcut1, inp.dcpt.sig1, inp.dcpt.nrad1, inp.dcpt.nang1, inp.dcpt.neighspe1,
            inp.dcpt.rep2, inp.dcpt.rcut2, inp.dcpt.sig2, inp.dcpt.nrad2, inp.dcpt.nang2, inp.dcpt.neighspe2,
            sparsify, inp.dcpt.sparsify.nsamples, inp.dcpt.sparsify.ncut,
            inp.gpr.z, inp.gpr.Menv, inp.gpr.Ntrain, inp.gpr.trainfrac, inp.gpr.regul, inp.gpr.eigcut,
            inp.gpr.gradtol, inp.gpr.restart, inp.gpr.blocksize, inp.gpr.trainsel
        )


    def check_input(self, inp:Dict):
        """Check keys (required, optional, not allowed), and value types and ranges

        Just check keys and values one by one, silly err haha"""

        """Format: (required, default value, value type, value extra check)
        About required:
            - True -> required
            - False -> optional, will fill in default value if not found
        """
        inp_template = {
            "salted": {
                "saltedname": (True, None, str, None),  # salted workflow identifier
                "saltedpath": (True, None, str, lambda inp, val: os.path.exists(val)),  # path to SALTED outputs
            },
            "sys": {
                "filename": (True, None, str, lambda inp, val: os.path.exists(val)),  # path to geometry file
                "species": (True, None, list, lambda inp, val: all(isinstance(i, str) for i in val)),  # list of species in all geometries
                "average": (False, True, bool, None),  # if bias the GPR by the average of predictions
                "field": (False, False, bool, None),  # if predict the field response
                "parallel": (False, False, bool, None),  # if use mpi4py
            },
            "qm": {
                "path2qm": (True, None, str, lambda inp, val: os.path.exists(val)),  # path to the QM calculation outputs
                "qmcode": (True, None, str, lambda inp, val: val.lower() in ('aims', 'pyscf', 'cp2k')),  # quantum mechanical code
                "qmbasis": (False, "PLACEHOLDER", str, lambda inp, val: (
                    ((inp["qm"]["qmcode"].lower() != 'pyscf') and (val == "PLACEHOLDER"))  # if not using pyscf, do not specify it
                    or
                    ((inp["qm"]["qmcode"].lower() == 'pyscf') and (val != "PLACEHOLDER"))  # if using pyscf, do specify it
                )),  # quantum mechanical basis, only for PySCF
                "functional": (False, "PLACEHOLDER", str, lambda inp, val: (
                    ((inp["qm"]["qmcode"].lower() != 'pyscf') and (val == "PLACEHOLDER"))  # if not using pyscf, do not specify it
                    or
                    ((inp["qm"]["qmcode"].lower() == 'pyscf') and (val != "PLACEHOLDER"))  # if using pyscf, do specify it
                )),  # quantum mechanical functional, only for PySCF
                "dfbasis": (True, None, str, None),  # density fitting basis
            },
            "pred": {
                "filename_pred": (True, None, str, lambda inp, val: os.path.exists(val)),  # path to the prediction file
                "predname": (True, None, str, None),  # SALTED prediction identifier
                "predict_data": (False, "aims_pred_data", str, None),  # path to the prediction data by QM code, for AIMS only
            },
            "dcpt": {
                "rep1": (True, None, str, lambda inp, val: val in ('rho', 'V')),  # descriptor, rho -> SOAP, V -> LODE
                "rcut1": (True, None, float, lambda inp, val: val > 0),  # cutoff radius
                "sig1": (True, None, float, lambda inp, val: val > 0),  # Gaussian width
                "nrad1": (True, None, int, lambda inp, val: val > 0),  # number of radial basis functions
                "nang1": (True, None, int, lambda inp, val: val > 0),  # number of angular basis functions
                "neighspe1": (True, None, list, lambda inp, val: all(isinstance(i, str) for i in val)),  # list of neighbor species
                "rep2": (True, None, str, lambda inp, val: val in ('rho', 'V')),
                "rcut2": (True, None, float, lambda inp, val: val > 0),
                "sig2": (True, None, float, lambda inp, val: val > 0),
                "nrad2": (True, None, int, lambda inp, val: val > 0),
                "nang2": (True, None, int, lambda inp, val: val > 0),
                "neighspe2": (True, None, list, lambda inp, val: all(isinstance(i, str) for i in val)),
                "sparsify": {
                    "nsamples": (False, 100, int, lambda inp, val: val > 0),  # number of samples for sparsifying feature channel
                    "ncut": (False, 0, int, lambda inp, val: (val == 0) or (val > 0)),  # number of features to keep
                }
            },
            "gpr": {
                "z": (False, 2.0, float, lambda inp, val: val > 0),  # kernel exponent
                "Menv": (True, None, int, lambda inp, val: val > 0),  # number of sparsified atomic environments
                "Ntrain": (True, None, int, lambda inp, val: val > 0),  # size of the training set (rest is validation set)
                "trainfrac": (False, 1.0, float, lambda inp, val: (val > 0) and (val <= 1.0)),  # fraction of the training set used for training
                "regul": (False, 1e-6, float, lambda inp, val: val > 0),  # regularisation parameter
                "eigcut": (False, 1e-10, float, lambda inp, val: val > 0),  # eigenvalues cutoff
                "gradtol": (False, 1e-5, float, lambda inp, val: val > 0),  # min gradient as stopping criterion for CG minimization
                "restart": (False, False, bool, lambda inp, val: isinstance(val, bool)),  # if restart the minimization
                "blocksize": (False, 100, int, lambda inp, val: val > 0),  # block size for matrix inversion
                "trainsel": (False, 'random', str, lambda inp, val: val in ('random', 'sequential')),  # if shuffle the training set
            }
        }

        def rec_applyPLACEHOLDER_vals(_inp, _inp_template):
            """apply default values if optional parameters are not found"""

            """check if the keys in inp exist in inp_template"""
            for key, val in _inp.items():
                if key not in _inp_template.keys():
                    raise ValueError(f"Key not allowed: {key}")
            """apply default values"""
            for key, val in _inp_template.items():
                if isinstance(val, dict):
                    """we can ignore a section if it's not required"""
                    if key not in _inp.keys():
                        _inp[key] = dict()  # make it an empty dict
                    _inp[key] = rec_applyPLACEHOLDER_vals(_inp[key], _inp_template[key])
                elif isinstance(val, tuple):
                    (required, valPLACEHOLDER, val_type, extra_check_func) = val
                    if key not in _inp.keys():
                        if required:
                            raise ValueError(f"Required key not found: {key}")
                        else:
                            _inp[key] = valPLACEHOLDER
                else:
                    raise ValueError(f"Invalid template value: {val}")
            return _inp

        def rec_check_vals(_inp, _inp_template):
            """check values' type and range"""
            for key, template in _inp_template.items():
                if isinstance(template, dict):
                    rec_check_vals(_inp[key], _inp_template[key])
                elif isinstance(template, tuple):
                    val = _inp[key]
                    (required, valPLACEHOLDER, val_type, extra_check_func) = _inp_template[key]
                    if not isinstance(val, val_type):
                        raise ValueError(f"Value type error: {key=}, {val=}, current_type={type(val)}, expected_type={val_type}")
                    if extra_check_func is not None:
                        if not extra_check_func(inp, val):
                            raise ValueError(f"Value check failed: {key=}, {val=}. Please check the required conditions.")
                else:
                    raise ValueError(f"Invalid template value: {template}")

        inp = rec_applyPLACEHOLDER_vals(inp, inp_template)  # now inp has all the keys as in inp_template in all levels
        rec_check_vals(inp, inp_template)

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

