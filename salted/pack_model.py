import h5py
import glob, os
import numpy as np

import struct

def u32(x):  return struct.pack('<I', x)
def i32(x):  return struct.pack('<i', x)
def i64(x):  return struct.pack('<q', x)
def f64(x):  return struct.pack('<d', x)

from salted.sys_utils import ParseConfig
#EVERYTHING IS LITTLE ENDIAN!

#FORMAT FOR SALTED FILE:
#MAGIC_NUMBER (5 bytes, str)
#VERSION (4 bytes, int32)
#N_BLOCKS (4 bytes, int32)
#FOR EACH BLOCK:
    #BLOCK_NAME (5 bytes, str)
    #BLOCK_LOCATION (4 bytes, int32)
#DATA (variable size)

#FORMAT FOR DATA: (Slightly depending on Context)
#TYPE_OF_DATA (4 bytes, int32)
#nfiles (4 bytes, int32) #If multiple files
#FOR EACH FILE:
    #ndims (4 bytes, int32)
    #dims (4*ndims bytes, int32)
    #data (ndims*8 bytes, float64)


types_dict = {
    "int32": 0,
    "int64": 1,
    "float32": 2,
    "float64": 3,
    "str": 4,
    "bool": 5,
}
OFFSET_TABLE_OF_CONTENTS = 5+4+4

def first_match(pattern):
    m = glob.glob(pattern)
    return m[0] if m else None

def write_key5(f, s: bytes):
    if len(s) > 5:
        raise ValueError(f"Key longer than 5 bytes: {s!r}")
    f.write(s + b'\0' * (5 - len(s)))

def write_data_head(SALTED_file, data):
    SALTED_file.write(i32(int(data.ndim)))
    for dim in data.shape:
        SALTED_file.write(i32(int(dim)))


def write_chunk_location(SALTED_file, chunk_name, location):
    global OFFSET_TABLE_OF_CONTENTS
    end_of_block = SALTED_file.tell()
    SALTED_file.seek(OFFSET_TABLE_OF_CONTENTS)
    write_key5(SALTED_file, chunk_name.encode("utf-8"))
    SALTED_file.write(i32(int(location)))
    SALTED_file.seek(end_of_block)
    OFFSET_TABLE_OF_CONTENTS += 5+4



#Format:
#TYPE_OF_DATA (4 bytes, int32)
#nfiles (4 bytes, int32)
#FOR EACH FILE:
    #element (5 bytes, str)
    #ndims (4 bytes, int32)
    #dims (4*ndims bytes, int32)
    #data (ndims*8 bytes, float64)
def pack_averages(SALTED_file, path):
    print("Writing Averages")
    begin_of_block = SALTED_file.tell()
    SALTED_file.write(i32(int(types_dict["float64"])))
    files = glob.glob(os.path.join(path,"averages",'av*.npy'))
    SALTED_file.write(i32(int(len(files))))
    for file in files:
        element = os.path.basename(file).split('.')[0].split("_")[-1]
        write_key5(SALTED_file, element.encode("utf-8"))
        data = np.load(file).astype(np.float64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(np.asarray(data,dtype='<f8').tobytes())
    write_chunk_location(SALTED_file, "AVERG", begin_of_block)
        

#HAS TO BE IN INCREASING ORDER
#Format:
#TYPE_OF_DATA (4 bytes, int32)
#nfiles (4 bytes, int32)
#FOR EACH FILE:
    #ndims (4 bytes, int32)
    #dims (4*ndims bytes, int32)
    #data (ndims*8 bytes, float64)
def pack_wigners(SALTED_file, path, inp):
    print("Writing Wigners")
    begin_of_block = SALTED_file.tell()
    SALTED_file.write(i32(int(types_dict["float64"])))
    files = glob.glob(os.path.join(path,"wigners",f'wigner_lam-*_lmax1-{inp.descriptor.rep1.nang}_lmax2-{inp.descriptor.rep2.nang}.dat'))
    files.sort(key=lambda x: int(x.split("_")[1].split("-")[1]))
    SALTED_file.write(i32(int(len(files))))
    for file in files:
        print(file)
        data = np.loadtxt(file).astype(np.float64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(np.asarray(data,dtype='<f8').tobytes())
    write_chunk_location(SALTED_file, "WIG", begin_of_block)


#LAMDA HAS TO BE IN INCREASING ORDER
#Format:
#TYPE_OF_DATA (4 bytes, int32)
#Nfiles (4 bytes, int32)
#FOR EACH FILE:
    #ndims (4 bytes, int32)
    #dims (4*ndims bytes, int32)
    #data (ndims*8 bytes, float64)
def pack_fps(SALTED_file, path, inp):
    print("Writing FPS")
    begin_of_block = SALTED_file.tell()
    SALTED_file.write(i32(int(types_dict["int64"])))
    files = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'fps{inp.descriptor.sparsify.ncut}-*.npy'))
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    SALTED_file.write(i32(int(len(files))))
    for file in files:
        print(file)
        data = np.load(file).astype(np.int64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(np.asarray(data,dtype='<i8').tobytes())
    write_chunk_location(SALTED_file, "FPS", begin_of_block)



#Format:
#TYPE_OF_DATA (4 bytes, int32)
#nkeys (4 bytes, int32)
#atomic_species (5 bytes, str)
#For each atomic species:
    #nlambda (4 bytes, int32)
    #FOR EACH lambda:
        #ndims (4 bytes, int32)
        #dims (4*ndims bytes, int32)
        #data (dim1*dim2*8 bytes, float64)
def pack_projectors(SALTED_file, path, inp):
    file_proj = first_match(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'projector_M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}.h5'))
    if file_proj is None:
        raise FileNotFoundError(f"No projector file found for M={inp.gpr.Menv}, zeta={inp.gpr.z} in {os.path.join(path, f'equirepr_{inp.salted.saltedname}')}")
    begin_of_block = SALTED_file.tell()
    print(f"Reading {file_proj}")
    SALTED_file.write(i32(int(types_dict["float64"])))
    with h5py.File(file_proj, 'r') as h5file:
        proje = h5file["projectors"]
        SALTED_file.write(i32(int(len(proje.keys()))))
        for key in sorted(proje.keys()):
            print(key, end=": ")
            #First 5 bytes are the key as string
            write_key5(SALTED_file, key.encode("utf-8"))
            #Write the number of sub-keys
            SALTED_file.write(i32(int(len(proje[key].keys()))))
            for key2 in sorted(proje[key].keys()):
                print(key2, end = " ")
                data = np.array(proje[key][key2], dtype=np.float64)
                write_data_head(SALTED_file, data)
                SALTED_file.write(np.asarray(data,dtype='<f8').tobytes())
            print()
    write_chunk_location(SALTED_file, "PROJE", begin_of_block)

def pack_FEATS(SALTED_file, path, inp):
    file_feat = first_match(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'FEAT_M-{inp.gpr.Menv}_*.h5'))
    if file_feat is None:
        raise FileNotFoundError(f"No FEAT file found for M={inp.gpr.Menv} in {os.path.join(path, f'equirepr_{inp.salted.saltedname}')}")
    begin_of_block = SALTED_file.tell()
    print(f"Reading {file_feat}")
    SALTED_file.write(i32(int(types_dict["float64"])))
    with h5py.File(file_feat, 'r') as h5file:
        descr = h5file["sparse_descriptors"]
        SALTED_file.write(i32(int(len(descr.keys()))))
        for key in sorted(descr.keys()):
            print(key, end=": ")
            #First 5 bytes are the key as string
            write_key5(SALTED_file, key.encode("utf-8"))
            #Write the number of sub-keys
            SALTED_file.write(i32(int(len(descr[key].keys()))))
            for key2 in sorted(descr[key].keys()):
                print(key2, end = " ")
                data = np.array(descr[key][key2], dtype=np.float64)
                write_data_head(SALTED_file, data)
                SALTED_file.write(np.asarray(data,dtype='<f8').tobytes())
            print()
    write_chunk_location(SALTED_file, "FEATS", begin_of_block)


#Format:
#TYPE_OF_DATA (4 bytes, int32)
#Nfiles (4 bytes, int32) Here 1
#FOR EACH FILE:
    #ndims (4 bytes, int32)
    #dims (4*ndims bytes, int32)
    #data (ndims*8 bytes, float64)
def pack_weights(SALTED_file, path, inp):
    begin_of_block = SALTED_file.tell()
    SALTED_file.write(i32(int(types_dict["float64"])))
    SALTED_file.write(i32(1))
    file = first_match(os.path.join(path, f"regrdir_{inp.salted.saltedname}", f"M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}", f'weights_N{int(inp.gpr.Ntrain*inp.gpr.trainfrac)}_reg*'))
    if file is None:
        raise FileNotFoundError(f"No weights file found for M={inp.gpr.Menv}, zeta={inp.gpr.z} in {os.path.join(path, f'regrdir_{inp.salted.saltedname}')}")
    data = np.load(file).astype(np.float64)
    write_data_head(SALTED_file, data)
    SALTED_file.write(np.asarray(data,dtype='<f8').tobytes())
    write_chunk_location(SALTED_file, "WEIGH", begin_of_block)

def pack_model_info(SALTED_file, inp):
    inputs = [
        (b"averg", inp.system.average),  #Bools
        (b"spars", inp.descriptor.sparsify.ncut > 0),
        (b"ncut\0", inp.descriptor.sparsify.ncut),  #int32
        (b"nang1", inp.descriptor.rep1.nang),
        (b"nang2", inp.descriptor.rep2.nang),
        (b"nrad1", inp.descriptor.rep1.nrad),
        (b"nrad2", inp.descriptor.rep2.nrad),
        (b"Menv\0", inp.gpr.Menv),
        (b"Ntran", inp.gpr.Ntrain),
        (b"rcut1", inp.descriptor.rep1.rcut),  #float64
        (b"rcut2", inp.descriptor.rep2.rcut),
        (b"sig1\0", inp.descriptor.rep1.sig),
        (b"sig2\0", inp.descriptor.rep2.sig),
        (b"zeta\0", inp.gpr.z),
        (b"trfra", inp.gpr.trainfrac),
        (b"speci", " ".join(inp.system.species)),   #str (and list str)
        (b"nspe1", " ".join(inp.descriptor.rep1.neighspe)),
        (b"nspe2", " ".join(inp.descriptor.rep2.neighspe)),
        (b"dfbas", inp.qm.dfbasis)
    ]
    begin_of_block = SALTED_file.tell()
    for key, value in inputs:
        print(key, value)
        write_key5(SALTED_file, key)
        if isinstance(value, bool):
            SALTED_file.write(i32(int(types_dict["bool"])))
            SALTED_file.write(i32(int(value)))
        elif isinstance(value, int):
            SALTED_file.write(i32(int(types_dict["int32"])))
            SALTED_file.write(i32(value))
        elif isinstance(value, float):
            SALTED_file.write(i32(int(types_dict["float64"])))
            SALTED_file.write(f64(value))
        elif isinstance(value, str):
            SALTED_file.write(i32(int(types_dict["str"])))
            enc = value.encode("utf-8")
            SALTED_file.write(i32(len(enc)))
            SALTED_file.write(enc)
        else:
            print(key, value)
            raise ValueError("Unknown type")
    write_chunk_location(SALTED_file, "CONFG", begin_of_block)


def preprocess_shells(basis):
    contractions_per_shell = []
    angular_momenta_per_shell = []
    coeffs_per_shell = []
    exponents_per_shell = []
    for shell in basis:
        angular_momenta_per_shell.append(shell[0])
        contractions_per_shell.append(len(shell) - 1)
        for exp,coef in shell[1:]:
            exponents_per_shell.append(exp)
            coeffs_per_shell.append(coef)
    return (contractions_per_shell, angular_momenta_per_shell,
            coeffs_per_shell, exponents_per_shell)


#Format:
#TYPE_OF_DATA (4 bytes, int32)
#nElements (4 bytes, int32)
#For each element:
    #Element ID (4 bytes, int32)
    #FOR EACH ARRAY iN [CONTRACTIONS, ANGULAR MOMENTA, EXPONENTS, COEFFS]:
        #N_DIMS (4 bytes, int32)
        #DIMS (nDims*4 bytes, int32)
        #DATA (dim1*8 bytes, float64)
def pack_basis(SALTED_file, inp):
    try:
        from pyscf import gto
        from pyscf.data.elements import ELEMENTS
        ELEMENTS_TO_NUM = {x: i for i,x in enumerate(ELEMENTS)}
    except ImportError:
        print("PySCF not found, cannot include basis sets in SALTED file, skipping basis packing")
        return
    
    begin_of_block = SALTED_file.tell()
    basis_name = inp.qm.dfbasis
    symbols = inp.system.species
    basis = {}
    for symbol in symbols:
        basis[symbol] = gto.basis.load(basis_name, symb=symbol)
    SALTED_file.write(i32(int(types_dict["float64"])))  # Data type
    SALTED_file.write(i32(int(len(symbols)))) # Number of elements (blocks to read after this)
    for elem in symbols:
        SALTED_file.write(i32(int(ELEMENTS_TO_NUM[elem])))
        shells = basis[elem]
        (contractions_per_shell,
         angular_momenta_per_shell,
         coeffs_per_shell,
         exponents_per_shell) = preprocess_shells(shells)
        write_data_head(SALTED_file, np.array(contractions_per_shell))
        SALTED_file.write(np.array(contractions_per_shell, dtype="<i4").tobytes())
        write_data_head(SALTED_file, np.array(angular_momenta_per_shell))
        SALTED_file.write(np.array(angular_momenta_per_shell, dtype="<i4").tobytes())
        write_data_head(SALTED_file, np.array(exponents_per_shell))
        SALTED_file.write(np.array(exponents_per_shell, dtype="<f8").tobytes())
        write_data_head(SALTED_file, np.array(coeffs_per_shell))
        SALTED_file.write(np.array(coeffs_per_shell, dtype="<f8").tobytes())

    write_chunk_location(SALTED_file, "BASIS", begin_of_block)

MAGIC_NUMBER = b"SALTD"  # 5-byte identifier
VERSION = 2
HAS_PYSCF = False
try:
    import pyscf
    HAS_PYSCF = True
except ImportError:
    HAS_PYSCF = False

BLOCKS = ["AVERG", "WIG", "FPS", "FEATS", "PROJE", "WEIGH", "CONFG"]
if HAS_PYSCF: BLOCKS.append("BASIS")
N_BLOCKS = len(BLOCKS)
def write_header(SALTED_file):
    SALTED_file.write(MAGIC_NUMBER)
    SALTED_file.write(i32(int(VERSION)))
    SALTED_file.write(i32(N_BLOCKS))
    #Leave (5+4)*N_Blocks bytes for the block names and locations (5bytes for the name, 4 bytes for the location)
    SALTED_file.write(b'\0'*(5+4)*N_BLOCKS)
    
def build():
    inp = ParseConfig().parse_input()
    path = inp.salted.saltedpath
    with open(f"{inp.salted.saltedname}.salted", "wb") as f:
        write_header(f)
        pack_model_info(f, inp)
        pack_averages(f, path)
        pack_wigners(f, path, inp)
        pack_fps(f, path, inp)
        pack_FEATS(f, path, inp)
        pack_projectors(f, path, inp)
        pack_weights(f, path, inp)
        if HAS_PYSCF:
            pack_basis(f, inp)

if __name__ == "__main__":
    build()