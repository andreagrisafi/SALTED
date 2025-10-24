import struct
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Unpacking functions for little-endian format
def read_u32(f): return struct.unpack('<I', f.read(4))[0]
def read_i32(f): return struct.unpack('<i', f.read(4))[0]
def read_i64(f): return struct.unpack('<q', f.read(8))[0]
def read_f64(f): return struct.unpack('<d', f.read(8))[0]
def read_bool(f): return struct.unpack('<?', f.read(1))[0]

types_dict = {
    0: "int32",
    1: "int64",
    2: "float32",
    3: "float64",
    4: "str",
    5: "bool",
}

MAGIC_NUMBER = b"SALTD"
SUPPORTED_VERSIONS = [2]

def read_key5(f) -> str:
    """Read a 5-byte key and strip null bytes"""
    key_bytes = f.read(5)
    return key_bytes.rstrip(b'\0').decode('utf-8')

def read_data_head(f) -> Tuple[int, Tuple[int, ...]]:
    """Read the dimensionality and shape of array data"""
    ndims = read_i32(f)
    dims = tuple(read_i32(f) for _ in range(ndims))
    return ndims, dims

def read_header(f) -> Tuple[int, Dict[str, int]]:
    """Read the file header and return version and block locations"""
    # Read magic number
    magic = f.read(5)
    if magic != MAGIC_NUMBER:
        raise ValueError(f"Invalid SALTED file: magic number is {magic!r}, expected {MAGIC_NUMBER!r}")
    
    # Read version
    version = read_i32(f)
    if version not in SUPPORTED_VERSIONS:
        raise ValueError(f"Unsupported version {version}, supported versions: {SUPPORTED_VERSIONS}")
    
    # Read number of blocks
    n_blocks = read_i32(f)
    
    # Read table of contents (block names and locations)
    blocks = {}
    for _ in range(n_blocks):
        block_name = read_key5(f)
        block_location = read_i32(f)
        blocks[block_name] = block_location
    
    return version, blocks

def read_averages(f) -> Dict[str, np.ndarray]:
    """Read the AVERG block containing averages for each element"""
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for averages, got {types_dict[data_type]}")
    
    nfiles = read_i32(f)
    averages = {}
    
    for _ in range(nfiles):
        element = read_key5(f)
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 8
        data = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
        averages[element] = data
    
    return averages

def read_wigners(f) -> List[np.ndarray]:
    """Read the WIG block containing Wigner matrices"""
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for wigners, got {types_dict[data_type]}")
    
    nfiles = read_i32(f)
    wigners = []
    
    for _ in range(nfiles):
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 8
        data = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
        wigners.append(data)
    
    return wigners

def read_fps(f) -> List[np.ndarray]:
    """Read the FPS block containing FPS indices"""
    data_type = read_i32(f)
    if types_dict[data_type] != "int64":
        raise ValueError(f"Expected int64 data type for FPS, got {types_dict[data_type]}")
    
    nfiles = read_i32(f)
    fps_data = []
    
    for _ in range(nfiles):
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 8
        data = np.frombuffer(f.read(nbytes), dtype='<i8').reshape(dims)
        fps_data.append(data)
    
    return fps_data

def read_projectors(f) -> Dict[str, Dict[str, np.ndarray]]:
    """Read the PROJE block containing projectors"""
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for projectors, got {types_dict[data_type]}")
    
    nkeys = read_i32(f)
    projectors = {}
    
    for _ in range(nkeys):
        species = read_key5(f)
        nlambda = read_i32(f)
        projectors[species] = {}
        
        for _ in range(nlambda):
            ndims, dims = read_data_head(f)
            nbytes = np.prod(dims) * 8
            data = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
            # Use the lambda index as key (0, 1, 2, ...)
            lambda_idx = len(projectors[species])
            projectors[species][str(lambda_idx)] = data
    
    return projectors

def read_feats(f) -> Dict[str, Dict[str, np.ndarray]]:
    """Read the FEATS block containing sparse descriptors"""
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for FEATS, got {types_dict[data_type]}")
    
    nkeys = read_i32(f)
    feats = {}
    
    for _ in range(nkeys):
        species = read_key5(f)
        nlambda = read_i32(f)
        feats[species] = {}
        
        for _ in range(nlambda):
            ndims, dims = read_data_head(f)
            nbytes = np.prod(dims) * 8
            data = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
            # Use the lambda index as key (0, 1, 2, ...)
            lambda_idx = len(feats[species])
            feats[species][str(lambda_idx)] = data
    
    return feats

def read_weights(f) -> np.ndarray:
    """Read the WEIGH block containing regression weights"""
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for weights, got {types_dict[data_type]}")
    
    nfiles = read_i32(f)
    if nfiles != 1:
        raise ValueError(f"Expected 1 weights file, got {nfiles}")
    
    ndims, dims = read_data_head(f)
    nbytes = np.prod(dims) * 8
    weights = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
    
    return weights

def read_config(f) -> Dict[str, Any]:
    """Read the CONFG block containing model configuration"""
    config = {}
    
    # Read until we hit the end of the block or another block marker
    # We need to read carefully since we don't know the block size
    start_pos = f.tell()
    
    try:
        while True:
            # Try to read a key
            pos_before = f.tell()
            key = read_key5(f)
            
            # Check if this might be a block marker (next block)
            # If we can't read a valid type, we've reached the end
            try:
                value_type = read_i32(f)
            except struct.error:
                f.seek(pos_before)
                break
            
            if value_type not in types_dict:
                # This is probably the start of the next block
                f.seek(pos_before)
                break
            
            type_name = types_dict[value_type]
            
            if type_name == "bool":
                value = bool(read_bool(f))
            elif type_name == "int32":
                value = read_i32(f)
            elif type_name == "float64":
                value = read_f64(f)
            elif type_name == "str":
                str_len = read_i32(f)
                value = f.read(str_len).decode('utf-8')
            else:
                raise ValueError(f"Unexpected type in config: {type_name}")
            
            config[key] = value
    except Exception as e:
        # If we encounter an error, we might be at the end
        pass
    
    return config

def read_basis(f) -> Dict[str, Dict[str, np.ndarray]]:
    """Read the BASIS block containing basis set information"""
    try:
        from pyscf.data.elements import ELEMENTS
    except ImportError:
        print("Warning: PySCF not found, element names won't be resolved")
        ELEMENTS = None
    
    data_type = read_i32(f)
    if types_dict[data_type] != "float64":
        raise ValueError(f"Expected float64 data type for basis, got {types_dict[data_type]}")
    
    n_elements = read_i32(f)
    basis = {}
    
    for _ in range(n_elements):
        element_num = read_i32(f)
        if ELEMENTS is not None and element_num < len(ELEMENTS):
            element_symbol = ELEMENTS[element_num]
        else:
            element_symbol = f"Element_{element_num}"
        
        basis[element_symbol] = {}
        
        # Read contractions per shell
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 4
        basis[element_symbol]['contractions'] = np.frombuffer(f.read(nbytes), dtype='<i4').reshape(dims)
        
        # Read angular momenta per shell
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 4
        basis[element_symbol]['angular_momenta'] = np.frombuffer(f.read(nbytes), dtype='<i4').reshape(dims)
        
        # Read exponents
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 8
        basis[element_symbol]['exponents'] = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
        
        # Read coefficients
        ndims, dims = read_data_head(f)
        nbytes = np.prod(dims) * 8
        basis[element_symbol]['coefficients'] = np.frombuffer(f.read(nbytes), dtype='<f8').reshape(dims)
    
    return basis

def read_salted_model(filename: str) -> Dict[str, Any]:
    """
    Read a SALTED model file and return all its contents.
    
    Parameters
    ----------
    filename : str
        Path to the .salted file
    
    Returns
    -------
    model : dict
        Dictionary containing all model data with keys:
        - 'version': File format version
        - 'config': Model configuration parameters
        - 'averages': Average values per element
        - 'wigners': Wigner matrices
        - 'fps': FPS indices
        - 'feats': Sparse descriptors
        - 'projectors': Projector matrices
        - 'weights': Regression weights
        - 'basis': Basis set information (if available)
    """
    model = {}
    
    with open(filename, 'rb') as f:
        # Read header and get block locations
        version, blocks = read_header(f)
        model['version'] = version
        
        print(f"SALTED model version {version}")
        print(f"Available blocks: {list(blocks.keys())}")
        
        # Read each block
        if 'CONFG' in blocks:
            print("Reading configuration...")
            f.seek(blocks['CONFG'])
            model['config'] = read_config(f)
        
        if 'AVERG' in blocks:
            print("Reading averages...")
            f.seek(blocks['AVERG'])
            model['averages'] = read_averages(f)
        
        if 'WIG' in blocks:
            print("Reading Wigner matrices...")
            f.seek(blocks['WIG'])
            model['wigners'] = read_wigners(f)
        
        if 'FPS' in blocks:
            print("Reading FPS indices...")
            f.seek(blocks['FPS'])
            model['fps'] = read_fps(f)
        
        if 'FEATS' in blocks:
            print("Reading sparse descriptors...")
            f.seek(blocks['FEATS'])
            model['feats'] = read_feats(f)
        
        if 'PROJ' in blocks:
            print("Reading projectors...")
            f.seek(blocks['PROJ'])
            model['projectors'] = read_projectors(f)
        
        if 'WEIGH' in blocks:
            print("Reading weights...")
            f.seek(blocks['WEIGH'])
            model['weights'] = read_weights(f)
        
        if 'BASIS' in blocks:
            print("Reading basis sets...")
            f.seek(blocks['BASIS'])
            model['basis'] = read_basis(f)
    
    return model

def print_model_summary(model: Dict[str, Any]):
    """Print a summary of the loaded model"""
    print("\n" + "="*60)
    print("SALTED Model Summary")
    print("="*60)
    
    print(f"\nVersion: {model.get('version', 'N/A')}")
    
    if 'config' in model:
        print("\nConfiguration:")
        for key, value in sorted(model['config'].items()):
            print(f"  {key}: {value}")
    
    if 'averages' in model:
        print(f"\nAverages: {len(model['averages'])} elements")
        for elem, data in model['averages'].items():
            print(f"  {elem}: shape {data.shape}")
    
    if 'wigners' in model:
        print(f"\nWigner matrices: {len(model['wigners'])} lambdas")
        for i, w in enumerate(model['wigners']):
            print(f"  Lambda {i}: shape {w.shape}")
    
    if 'fps' in model:
        print(f"\nFPS indices: {len(model['fps'])} arrays")
        for i, fps in enumerate(model['fps']):
            print(f"  Array {i}: shape {fps.shape}")
    
    if 'feats' in model:
        print(f"\nSparse descriptors: {len(model['feats'])} species")
        for species, lambdas in model['feats'].items():
            print(f"  {species}: {len(lambdas)} lambdas")
    
    if 'projectors' in model:
        print(f"\nProjectors: {len(model['projectors'])} species")
        for species, lambdas in model['projectors'].items():
            print(f"  {species}: {len(lambdas)} lambdas")
    
    if 'weights' in model:
        print(f"\nWeights: shape {model['weights'].shape}")
    
    if 'basis' in model:
        print(f"\nBasis sets: {len(model['basis'])} elements")
        for elem in model['basis'].keys():
            print(f"  {elem}")
    
    print("="*60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python read_model.py <model_file.salted>")
        sys.exit(1)
    
    filename = sys.argv[1]
    print(f"Reading SALTED model from {filename}...")
    
    try:
        model = read_salted_model(filename)
        print_model_summary(model)
    except Exception as e:
        print(f"Error reading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
