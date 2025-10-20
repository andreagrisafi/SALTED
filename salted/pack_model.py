import h5py
import glob, os
import numpy as np

from salted.sys_utils import ParseConfig
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


def write_data_head(SALTED_file, data):
    SALTED_file.write(np.int32(data.ndim))
    for dim in data.shape:
        SALTED_file.write(np.int32(dim))



def write_chunk_location(SALTED_file, chunk_name, location):
    global OFFSET_TABLE_OF_CONTENTS
    end_of_block = SALTED_file.tell()
    SALTED_file.seek(OFFSET_TABLE_OF_CONTENTS)
    SALTED_file.write(chunk_name.encode("utf-8"))
    SALTED_file.write(b'\0'*(5-len(chunk_name)))
    SALTED_file.write(np.int32(location).tobytes())
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
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
    files = glob.glob(os.path.join(path,"averages",'av*.npy'))
    SALTED_file.write(np.int32(len(files)).tobytes())
    for file in files:
        element = os.path.basename(file).split('.')[0].split("_")[-1]
        SALTED_file.write(element.encode("utf-8"))
        SALTED_file.write(b'\0'*(5-len(element)))
        data = np.load(file).astype(np.float64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(data.tobytes())
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
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
    files = glob.glob(os.path.join(path,"wigners",f'wigner_lam-*_lmax1-{inp.descriptor.rep1.nang}_lmax2-{inp.descriptor.rep2.nang}.dat'))
    files.sort(key=lambda x: int(x.split("_")[1].split("-")[1]))
    SALTED_file.write(np.int32(len(files)).tobytes())
    for file in files:
        print(file)
        data = np.loadtxt(file).astype(np.float64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(data.tobytes())
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
    SALTED_file.write(np.int32(types_dict["int64"]).tobytes())
    files = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'fps{inp.descriptor.sparsify.ncut}-*.npy'))
    files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
    SALTED_file.write(np.int32(len(files)).tobytes())
    for file in files:
        print(file)
        data = np.load(file).astype(np.int64)
        write_data_head(SALTED_file, data)
        SALTED_file.write(data.tobytes())
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
    file_proj = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'projector_M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}.h5'))[0]
    begin_of_block = SALTED_file.tell()
    print(f"Reading {file_proj}")
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
    with h5py.File(file_proj, 'r') as h5file:
        SALTED_file.write(np.int32(len(h5file["projectors"].keys())).tobytes())
        for key in list(h5file["projectors"].keys()):
            print(key, end=": ")
            #First 5 bytes are the key as string
            SALTED_file.write(key.encode("utf-8"))
            SALTED_file.write(b'\0'*(5-len(key)))
            #Write the number of sub-keys
            SALTED_file.write(np.int32(len(h5file["projectors"][key].keys())).tobytes())
            for key2 in h5file["projectors"][key].keys():
                print(key2, end = " ")
                data = np.array(h5file["projectors"][key][key2], dtype=np.float64)
                write_data_head(SALTED_file, data)
                SALTED_file.write(data.tobytes())
            print()
    write_chunk_location(SALTED_file, "PROJ", begin_of_block)

def pack_FEATS(SALTED_file, path, inp):
    file_feat = glob.glob(os.path.join(path, f"equirepr_{inp.salted.saltedname}", f'FEAT_M-{inp.gpr.Menv}.h5'))[0]
    begin_of_block = SALTED_file.tell()
    print(f"Reading {file_feat}")
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
    with h5py.File(file_feat, 'r') as h5file:
        SALTED_file.write(np.int32(len(h5file["sparse_descriptors"].keys())).tobytes())
        for key in list(h5file["sparse_descriptors"].keys()):
            print(key, end=": ")
            #First 5 bytes are the key as string
            SALTED_file.write(key.encode("utf-8"))
            SALTED_file.write(b'\0'*(5-len(key)))
            #Write the number of sub-keys
            SALTED_file.write(np.int32(len(h5file["sparse_descriptors"][key].keys())).tobytes())
            for key2 in h5file["sparse_descriptors"][key].keys():
                print(key2, end = " ")
                data = np.array(h5file["sparse_descriptors"][key][key2], dtype=np.float64)
                write_data_head(SALTED_file, data)
                SALTED_file.write(data.tobytes())
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
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
    SALTED_file.write(np.int32(1).tobytes())
    file = glob.glob(os.path.join(path, f"regrdir_{inp.salted.saltedname}", f"M{inp.gpr.Menv}_zeta{inp.gpr.z:.1f}", f'weights_N{int(inp.gpr.Ntrain*inp.gpr.trainfrac)}_reg*'))[0]
    data = np.load(file).astype(np.float64)
    write_data_head(SALTED_file, data)
    SALTED_file.write(data.tobytes())
    write_chunk_location(SALTED_file, "WEIGH", begin_of_block)

def pack_model_info(SALTED_file, inp):
    inputs = {
        b"averg": inp.system.average,  #Bools
        b"spars": inp.descriptor.sparsify.ncut > 0,
        b"ncut\0": inp.descriptor.sparsify.ncut,  #int32
        b"nang1": inp.descriptor.rep1.nang,
        b"nang2": inp.descriptor.rep2.nang,
        b"nrad1": inp.descriptor.rep1.nrad,
        b"nrad2": inp.descriptor.rep2.nrad,
        b"Menv\0": inp.gpr.Menv,
        b"Ntran": inp.gpr.Ntrain,
        b"rcut1": inp.descriptor.rep1.rcut,  #float64
        b"rcut2": inp.descriptor.rep2.rcut,
        b"sig1\0": inp.descriptor.rep1.sig,
        b"sig2\0": inp.descriptor.rep2.sig,
        b"zeta\0": inp.gpr.z,
        b"trfra": inp.gpr.trainfrac,
        b"speci": " ".join(inp.system.species),   #str (and list str)
        b"nspe1": " ".join(inp.descriptor.rep1.neighspe),
        b"nspe2": " ".join(inp.descriptor.rep2.neighspe),
        b"dfbas": inp.qm.dfbasis
    }
    begin_of_block = SALTED_file.tell()
    for key in inputs.keys():
        print(key, inputs[key])
        if isinstance(inputs[key], bool):
            SALTED_file.write(key)
            SALTED_file.write(np.int32(types_dict["bool"]).tobytes())
            SALTED_file.write(np.bool_(int(inputs[key])).tobytes())
        elif isinstance(inputs[key], int):
            SALTED_file.write(key)
            SALTED_file.write(np.int32(types_dict["int32"]).tobytes())
            SALTED_file.write(np.int32(inputs[key]).tobytes())
        elif isinstance(inputs[key], float):
            SALTED_file.write(key)
            SALTED_file.write(np.int32(types_dict["float64"]).tobytes())
            SALTED_file.write(np.float64(inputs[key]).tobytes())
        elif isinstance(inputs[key], str):
            SALTED_file.write(key)
            SALTED_file.write(np.int32(types_dict["str"]).tobytes())
            SALTED_file.write(np.int32(len(inputs[key])).tobytes())
            print(inputs[key].encode("utf-8"))
            SALTED_file.write(inputs[key].encode("utf-8"))
        else:
            print(key, inputs[key])
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
    #nShells (4 bytes, int32)
    #nPrimitives (4 bytes, int32)
    #contractions_per_shell (nShells*4 bytes, int32)
    #angular_momenta_per_shell (nShells*4 bytes, int32)
    #exponents_per_shell (nPrimitives*8 bytes, float64)
    #coeffs_per_shell (nPrimitives*8 bytes, float64)
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
    SALTED_file.write(np.int32(types_dict["float64"]).tobytes())  # Data type
    SALTED_file.write(np.int32(len(symbols)).tobytes()) # Number of elements (blocks to read after this)
    for elem in symbols:
        SALTED_file.write(np.int32(ELEMENTS_TO_NUM[elem]).tobytes())
        shells = basis[elem]
        (contractions_per_shell,
         angular_momenta_per_shell,
         coeffs_per_shell,
         exponents_per_shell) = preprocess_shells(shells)
        SALTED_file.write(np.int32(len(shells)).tobytes())  # Number of shells
        SALTED_file.write(np.int32(len(coeffs_per_shell)).tobytes())  # Number of primitives
        SALTED_file.write(np.array(contractions_per_shell, dtype=np.int32).tobytes())
        SALTED_file.write(np.array(angular_momenta_per_shell, dtype=np.int32).tobytes())
        SALTED_file.write(np.array(exponents_per_shell, dtype=np.float64).tobytes())
        SALTED_file.write(np.array(coeffs_per_shell, dtype=np.float64).tobytes())
    
    write_chunk_location(SALTED_file, "BASIS", begin_of_block)

MAGIC_NUMBER = b"SALTD"  # 5-byte identifier
VERSION = np.int32(2)
BLOCKS = ["AVERG", "WIG", "FPS", "DESCR", "PROJE", "WEIGH", "CONF", "BASIS"]
N_BLOCKS = np.int32(len(BLOCKS))
def write_header(SALTED_file):
    SALTED_file.write(MAGIC_NUMBER)
    SALTED_file.write(VERSION.tobytes())
    SALTED_file.write(N_BLOCKS.tobytes())
    #Leave (5*4)*N_Blocks bytes for the block names and locations (5bytes for the name, 4 bytes for the location)
    SALTED_file.write(b'\0'*(5*4)*N_BLOCKS)
    
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
        pack_basis(f, inp)

if __name__ == "__main__":
    build()