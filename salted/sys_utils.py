import os
import re
import sys
from typing import Any, Callable, Dict, Optional, Tuple, Union, Literal, List

import numpy as np
import yaml
from ase.io import read

from salted import basis



def read_system(filename:str=None, spelist:List[str]=None, dfbasis:str=None):

    if (filename is None) and (spelist is None) and (dfbasis is None):
        inp = ParseConfig().parse_input()
        filename = inp.system.filename
        spelist = inp.system.species
        dfbasis = inp.qm.dfbasis
    elif (filename is not None) and (spelist is not None) and (dfbasis is not None):
        pass
    else:
        raise ValueError("Invalid input, should be either all None or all not None")

    # read basis
    [lmax,nmax] = basis.basiset(dfbasis)
    llist = []
    nlist = []
    for spe in spelist:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    nnmax = max(nlist)
    llmax = max(llist)

    # read system
    xyzfile = read(filename, ":", parallel=False)
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


ARGHELP_INDEX_STR = """Indexes to calculate, start from 0. Format: 1,3-5,7-10. Default is "all", which means all structures."""

def parse_index_str(index_str:Union[str, Literal["all"]]) -> Union[None, Tuple]:
    """Parse index string, e.g. "1,3-5,7-10" -> (1,3,4,5,7,8,9,10)

    If index_str is "all", return None. (indicating all structures)
    If index_str is "1,3-5,7-10", return (1,3,4,5,7,8,9,10)
    """

    if index_str == "all":
        return None
    else:
        assert isinstance(index_str, str)
        indexes = []
        for s in index_str.split(","):  # e.g. ["1", "3-5", "7-10"]
            assert all([c.isdigit() or c == "-" for c in s]), f"Invalid index format: {s}"
            if "-" in s:
                assert s.count("-") == 1, f"Invalid index format: {s}"
                start, end = s.split("-")
                assert start.isdigit() and end.isdigit, f"Invalid index format: {s}"
                indexes.extend(range(int(start), int(end) + 1))
            elif s.isdigit():
                indexes.append(int(s))
            else:
                raise ValueError(f"Invalid index format: {s}")
        return tuple(indexes)


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


PLACEHOLDER = "__PLACEHOLDER__"

class ParseConfig:
    """Input configuration file parser

    To use it, make sure an `inp.yaml` file exists in the current working directory,
    and simply run `ParseConfig().parse_input()`.
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
         zeta, Menv, Ntrain, trainfrac, regul, eigcut,
         gradtol, restart, blocksize, trainsel) = ParseConfig().get_all_params()
        ```
        """
        inp = self.parse_input()
        sparsify = False if inp.descriptor.sparsify.ncut == 0 else True
        return (
            inp.salted.saltedname, inp.salted.saltedpath,
            inp.system.filename, inp.system.species, inp.system.average, inp.system.field, inp.system.parallel,
            inp.qm.path2qm, inp.qm.qmcode, inp.qm.qmbasis, inp.qm.dfbasis,
            inp.prediction.filename_pred, inp.prediction.predname, inp.prediction.predict_data,
            inp.descriptor.rep1.type, inp.descriptor.rep1.rcut, inp.descriptor.rep1.sig,
            inp.descriptor.rep1.nrad, inp.descriptor.rep1.nang, inp.descriptor.rep1.neighspe,
            inp.descriptor.rep2.type, inp.descriptor.rep2.rcut, inp.descriptor.rep2.sig,
            inp.descriptor.rep2.nrad, inp.descriptor.rep2.nang, inp.descriptor.rep2.neighspe,
            sparsify, inp.descriptor.sparsify.nsamples, inp.descriptor.sparsify.ncut,
            inp.gpr.z, inp.gpr.Menv, inp.gpr.Ntrain, inp.gpr.trainfrac, inp.gpr.regul, inp.gpr.eigcut,
            inp.gpr.gradtol, inp.gpr.restart, inp.gpr.blocksize, inp.gpr.trainsel
        )

    def get_all_params_simple1(self) -> Tuple:
        """return all parameters with a tuple

        Please copy & paste:
        ```python
        (
            filename, species, average, field, parallel,
            rep1, rcut1, sig1, nrad1, nang1, neighspe1,
            rep2, rcut2, sig2, nrad2, nang2, neighspe2,
            sparsify, nsamples, ncut,
            z, Menv, Ntrain, trainfrac, regul, eigcut,
            gradtol, restart, blocksize, trainsel
        ) = ParseConfig().get_all_params_simple1()
        ```
        """
        inp = self.parse_input()
        sparsify = False if inp.descriptor.sparsify.ncut == 0 else True
        return (
            inp.system.filename, inp.system.species, inp.system.average, inp.system.field, inp.system.parallel,
            inp.descriptor.rep1.type, inp.descriptor.rep1.rcut, inp.descriptor.rep1.sig,
            inp.descriptor.rep1.nrad, inp.descriptor.rep1.nang, inp.descriptor.rep1.neighspe,
            inp.descriptor.rep2.type, inp.descriptor.rep2.rcut, inp.descriptor.rep2.sig,
            inp.descriptor.rep2.nrad, inp.descriptor.rep2.nang, inp.descriptor.rep2.neighspe,
            sparsify, inp.descriptor.sparsify.nsamples, inp.descriptor.sparsify.ncut,
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
            - False + PLACEHOLDER -> optional in some cases, but required in others cases
            - (if the default value is $PLACEHOLDER, it means the key is optional for some cases, but required for others)

        About PLACEHOLDER:
            - If a key is optional in some cases, but required in others, the default value is set to PLACEHOLDER.
            - The extra value checking should consider the PLACEHOLDER value!
        """

        rep_template = {
            "type": (True, None, str, lambda inp, val: val in ('rho', 'V')),  # descriptor, rho -> SOAP, V -> LODE
            "rcut": (True, None, float, lambda inp, val: val > 0),  # cutoff radius
            "sig": (True, None, float, lambda inp, val: val > 0),  # Gaussian width
            "nrad": (True, None, int, lambda inp, val: val > 0),  # number of radial basis functions
            "nang": (True, None, int, lambda inp, val: val > 0),  # number of angular basis functions
            "neighspe": (True, None, list, lambda inp, val: (
                all(isinstance(i, str) for i in val)
                # and
                # all(i in inp["system"]["species"] for i in val)  # species might be a subset of neighspe in Andrea's application
            )),  # list of neighbor species
        }
        inp_template = {
            "salted": {
                "saltedname": (True, None, str, None),  # salted workflow identifier
                "saltedpath": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to SALTED outputs / working directory
            },
            "system": {
                "filename": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to geometry file (training set)
                "species": (True, None, list, lambda inp, val: all(isinstance(i, str) for i in val)),  # list of species in all geometries
                "average": (False, True, bool, None),  # if bias the GPR by the average of predictions
                "field": (False, False, bool, None),  # if predict the field response
                "parallel": (False, False, bool, None),  # if use mpi4py
                "seed": (False, 42, int, None),  # random seed
            },
            "qm": {
                "path2qm": (True, None, str, lambda inp, val: check_path_exists(val)),  # path to the QM calculation outputs
                "qmcode": (True, None, str, lambda inp, val: val.lower() in ('aims', 'pyscf', 'cp2k')),  # quantum mechanical code
                "dfbasis": (True, None, str, None),  # density fitting basis
                #### below are optional, but required for some qmcode ####
                "qmbasis": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "pyscf")),  # quantum mechanical basis, only for PySCF
                "functional": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "pyscf")),  # quantum mechanical functional, only for PySCF
                "pseudocharge": (False, PLACEHOLDER, float, lambda inp, val: entry_with_qmcode(inp, val, "cp2k")),  # pseudo nuclear charge, only for CP2K
                "coeffile": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "cp2k")),
                "ovlpfile": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "cp2k")),
                "periodic": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "cp2k")),  # periodic boundary conditions, only for CP2K
            },
            "prediction": {
                "filename_pred": (False, PLACEHOLDER, str, lambda inp, val: check_path_exists(val)),  # path to the prediction file
                "predname": (False, PLACEHOLDER, str, None),  # SALTED prediction identifier
                #### below are optional, but required for some qmcode ####
                "predict_data": (False, PLACEHOLDER, str, lambda inp, val: entry_with_qmcode(inp, val, "aims")),  # path to the prediction data by QM code, only for AIMS
            },
            "descriptor": {
                "rep1": rep_template,  # descriptor 1
                "rep2": rep_template,  # descriptor 2
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
                "blocksize": (False, 0, int, lambda inp, val: val >= 0),  # block size for matrix inversion
                "trainsel": (False, 'random', str, lambda inp, val: val in ('random', 'sequential')),  # if shuffle the training set
            }
        }

        def rec_apply_default_vals(_inp, _inp_template, _prev_key:str):
            """apply default values if optional parameters are not found"""

            """check if the keys in inp exist in inp_template"""
            for key, val in _inp.items():
                if key not in _inp_template.keys():
                    raise ValueError(f"Key not allowed: {_prev_key+key}")
            """apply default values"""
            for key, val in _inp_template.items():
                if isinstance(val, dict):
                    """we can ignore a section if it's not required"""
                    if key not in _inp.keys():
                        _inp[key] = dict()  # make it an empty dict
                    _inp[key] = rec_apply_default_vals(_inp[key], _inp_template[key], _prev_key+key+".")
                elif isinstance(val, tuple):
                    (required, val_default, val_type, extra_check_func) = val
                    if key not in _inp.keys():
                        if required:
                            raise ValueError(f"Required key not found: {_prev_key+key}")
                        else:
                            _inp[key] = val_default
                else:
                    raise ValueError(f"Invalid template value: {val}")
            return _inp

        def rec_check_vals(_inp, _inp_template, _prev_key:str):
            """check values' type and range"""
            for key, template in _inp_template.items():
                if isinstance(template, dict):
                    rec_check_vals(_inp[key], _inp_template[key], _prev_key+key+".")
                elif isinstance(template, tuple):
                    val = _inp[key]
                    (required, val_default, val_type, extra_check_func) = _inp_template[key]
                    """There are cases that a value is required for certain conditions, so we always need to run extra_check_func"""
                    if (not isinstance(val, val_type)) and (val != PLACEHOLDER):  # if is PLACEHOLDER, then don't check the type
                        raise ValueError(f"Value type error: key={_prev_key+key}, {val=}, current_type={type(val)}, expected_type={val_type}")
                    if extra_check_func is not None:  # always run extra_check_func if not None
                        if not extra_check_func(inp, val):
                            raise ValueError(f"Value check failed: key={_prev_key+key}, {val=}. Please check the required conditions.")
                else:
                    raise ValueError(f"Invalid template value: {template}")

        inp = rec_apply_default_vals(inp, inp_template, "")  # now inp has all the keys as in inp_template in all levels
        rec_check_vals(inp, inp_template, "")

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



def entry_with_qmcode(inp, val, qmcode:Union[str, List[str]]) -> bool:
    """This means the entry is required IF and ONLY IF when using a specific qmcode"""
    if isinstance(qmcode, str):
        qmcode = [qmcode]
    return (
        ((inp["qm"]["qmcode"].lower() not in qmcode) and (val == PLACEHOLDER))  # if not using this qmcode, do not specify it
        or
        ((inp["qm"]["qmcode"].lower() in qmcode) and (val != PLACEHOLDER))  # if using this qmcode, do specify it
    )

def check_path_exists(path:str) -> bool:
    """Check if the path exists, the path should be either absolute or relative to the current working directory"""
    # return True  # for testing only
    if path == PLACEHOLDER:
        return True
    else:
        return os.path.exists(path)



class Irreps(tuple):
    """Handle irreducible representation arrays, like slices, multiplicities, etc."""

    def __new__(cls, irreps:Union[str, List[int], Tuple[int]]) -> 'Irreps':
        """
        irreps:
            - str, e.g. `1x0+2x1+3x2+3x3+2x4+1x5`
            - Tuple[Tuple[int]], e.g. ((1, 0), (2, 1), (3, 2), (3, 3), (2, 4), (1, 5),)
                - The super() tuple info
            - Tuple[int], e.g. (0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5,)
            - internal representation: the same as Tuple[Tuple[int]]
        """
        if isinstance(irreps, str):
            irreps_info_split = tuple(sec.strip() for sec in irreps.split("+") if len(sec) > 0)  # ("1x0", "2x1", ...)
            mul_l_tuple = tuple(  # ((1, 0), (2, 1), ...)
                tuple(int(i.strip()) for i in sec.split("x"))
                for sec in irreps_info_split
            )
            return super().__new__(cls, mul_l_tuple)
        elif isinstance(irreps, list) or isinstance(irreps, tuple):
            if len(irreps) == 0:
                return super().__new__(cls, ())
            elif isinstance(irreps[0], tuple) or isinstance(irreps[0], list):
                assert all(
                    all(isinstance(i, int) for i in mul_l) and len(mul_l) == 2 and mul_l[0] >= 0 and mul_l[1] >= 0
                    for mul_l in irreps
                ), ValueError(f"Invalid irreps_info: {irreps}")
                return super().__new__(cls, tuple(tuple(mul_l) for mul_l in irreps))
            elif isinstance(irreps[0], int):
                assert all(isinstance(i, int) and i >= 0 for i in irreps), ValueError(f"Invalid irreps_info: {irreps}")
                this_l_cnt, this_l = 1, irreps[0]
                mul_l_list:List[Tuple[int]] = []
                for l in irreps[1:]:
                    if l == this_l:
                        this_l_cnt += 1
                    else:
                        mul_l_list.append((this_l_cnt, this_l))
                        this_l_cnt, this_l = 1, l
                mul_l_list.append((this_l_cnt, this_l))
                print(mul_l_list)
                return super().__new__(cls, tuple(mul_l_list))
            else:
                raise ValueError(f"Invalid irreps_info: {irreps}")
        else:
            raise ValueError(f"Invalid irreps_info: {irreps}")

    @property
    def dim(self):
        """total dimension / length by magnetic quantum number"""
        return sum(mul * (2*l + 1) for mul, l in self)

    @property
    def num_irreps(self):
        """number of irreps, the sum of multiplicities of each l"""
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        """list of l values in the irreps"""
        return tuple(l for mul, l in self for _ in range(mul))

    @property
    def lmax(self) -> int:
        """maximum l in the irreps"""
        return max(tuple(l for _, l in self))

    def __repr__(self):
        return "+".join(f"{mul}x{l}" for mul, l in self)

    def __add__(self, other: 'Irreps') -> 'Irreps':
        return Irreps(super().__add__(other))

    def slices(self) -> List[slice]:
        """return all the slices for each l"""
        if hasattr(self, "_slices"):
            return self._slices
        else:
            self._slices = []
            ls = self.ls
            l_m_nums = tuple(2*l + 1 for l in ls)
            pointer = 0
            for m_num in l_m_nums:
                self._slices.append(slice(pointer, pointer+m_num))
                pointer += m_num
            assert pointer == self.dim
            self._slices = tuple(self._slices)
        return self._slices

    def slices_l(self, l:int) -> List[slice]:
        """return all the slices for a specific l"""
        return tuple(sl for _l, sl in zip(self.ls, self.slices()) if l == _l)

    def simplify(self) -> 'Irreps':
        """sort by l and combine the same l"""
        uniq_ls = tuple(set(self.ls))
        mul_ls = tuple((self.ls.count(l), l) for l in uniq_ls)
        return Irreps(mul_ls)

    def sort(self) -> 'Irreps':
        """"""
        raise NotImplementedError
