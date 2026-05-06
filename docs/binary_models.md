# Binary Salted Models
By running `salted_pack.py`, a single binary file is created that consolidates and serializes multiple data sources (NumPy arrays, HDF5 datasets, and model parameters) into one portable format.
This simplifies model sharing and enables easier deployment for prediction tasks.

## Usage
### Creating binary models
Only a single command is required to transform a newly trained and verified model into the binary `.salted` format.
The command must be executed from the main project directory:

    salted_pack.py

This will generate a `.salted` file.

### Deploying binary models
Predictions using a binary model can be performed with:

    predict_from_model.py

**Input arguments**
| Argument  | Meaning |
| -------- | --------|
| model | Path to the `.salted` model file.  |
| xyz   | Path to the `.xyz` file containing the structure for which the prediction is performed.  |
| -o     | Output directory for the predicted coefficients (optional, default: `predictions/`)  | 

## Implementation details
### Basic design
All integers are little-endian signed 32-bit unless noted; all floating arrays are little-endian float64. All 5-char names are ASCII padded with NUL to 5 bytes.
General Arrays

In the following every array is encoded using the same scheme:

    def encode_array(NDIMS:int, DIMS:list[int], data):
      file.write(NDIMS)
      for dim_size in DIMS:
        file.write(dim_size)
      file.write(data)

This results in a datastructure as follows:

    NDIMS (int32)
    DIMS (int32 * NDIMS)
    DATA (TYPE OF ARRAY)

The Type of the given array is encoded using the numbers from 0 - 5:

    int32=0
    int64=1
    float32=2
    float64=3
    str=4
    bool=5

### Sections in the file
Now the different sections of the file are explained in more detail:
#### Container header

    MAGIC (5 bytes): b"SALTD"
    VERSION (int32)
    N_BLOCKS (int32)
    TOC: N_BLOCKS entries of:
        BLOCK_NAME (5 bytes, NUL-padded)
        BLOCK_OFFSET (int32): file offset (from start) where the block payload begins

#### AVERG, WIG, FPS, WEIGH

    TYPE (int32) (datatype of the following block)
    NFILES (int32) (number of arrays in the specific key)
    FOR EACH FILE
        encode_array(NDIMS, DIMS, data)

#### FEATS, PROJE

    TYPE (int32): float64
    NKEYS (int32) — top-level HDF5 group count (sorted)
    For each top-level key:
        KEY5 (5 bytes, NUL-padded)
        NSUB (int32) — number of datasets under this key (sorted)
        For each sub-key dataset:
            encode_array(NDIMS, DIMS, data)

#### CONFG

    For each entry in inputs (fixed order in code):
        KEY5 (5 bytes) — e.g., b"averg", b"ncut\0", …
        TYPE (int32)
        VALUE encoded by VAL_TYPE:
            bool: int32 (0 or 1)
            int32: int32
            float64: float64
            str: SLEN (int32, byte length), then SLEN bytes UTF-8

#### BASIS (Only if pyscf is installed)

    TYPE (int32): float64 (tag for numeric arrays in this block)
    NELEM (int32): number of elements included
    For each element:
        ELEM_ID (int32): PySCF element index
        Four arrays follow, each preceded by a shape header:
            contractions_per_shell (int32[])
                encode_array(NDIMS, DIMS, data)
            angular_momenta_per_shell (int32[])
                encode_array(NDIMS, DIMS, data)
            exponents_per_shell (float64[])
                encode_array(NDIMS, DIMS, data)
            coeffs_per_shell (float64[])
                encode_array(NDIMS, DIMS, data)
