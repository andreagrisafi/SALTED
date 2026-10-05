"""
Dense-block computation of the regression matrices in hessian_matrix

Used when `sparse_algorithm: dense_blocks` is set in the `gpr` section of the input file.

For each training structure, hessian_matrix accumulates Psi^T @ S @ Psi, where S is the
overlap matrix of the density-fitting basis and Psi the RKHS vectors of the structure.

The non-zero entries of Psi form a "dense blocks" for each species and angular momentum l
(atoms x sparse environments), repeated for every radial function n. This module computes
Psi^T @ S @ Psi from these blocks with dense matrix products (BLAS, through numpy and
scipy), adding the result into the regression matrix.

The density-response case is not supported yet.

The number of threads is set through the BLAS library, e.g. OPENBLAS_NUM_THREADS.

Setting the environment variable SALTED_DENSE_BLOCKS_CHECK=1 also verifies that the block
is identical for every radial function (slower, meant for testing).
"""

import os
import time

import numpy as np
from scipy.linalg.blas import dgemm


class LayoutMismatch(Exception):
    """psi does not have the block structure expected."""


class BlockLayout:
    """Column layout of psi, the same for all configurations.
    """

    def __init__(self, species, lmax, nmax, widths):
        self.species = list(species)
        self.lmax = lmax
        self.nmax = nmax
        self.widths = widths
        self.col0 = {}  # (spe, l) -> first column of channel (spe, l, n=0)
        self.spe_cols = {}  # spe -> (first column, last column + 1) of the species
        # Loop over the columns of psi in the order rkhs_vector.py writes them:
        # species -> l -> n, each channel (spe, l, n) taking widths[(spe, l)] columns.
        # k is the "next free" column.
        k = 0
        for spe in self.species:
            first = k # first column of the species
            for l in range(lmax[spe] + 1):
                self.col0[(spe, l)] = k # first column of channel (spe, l, n=0)
                k += nmax[(spe, l)] * widths[(spe, l)] # skip the nmax copies of the table
            self.spe_cols[spe] = (first, k)
        self.ncols = k # total number of columns, to be compared with psi.shape[1]

    @classmethod
    def from_projectors(cls, path, species, lmax, nmax):
        """Read M_{s,l} from the dataset shapes of the RKHS projector file (no data read)."""
        import h5py

        widths = {}
        with h5py.File(path, "r") as f:
            for spe in species:
                for l in range(lmax[spe] + 1):
                    widths[(spe, l)] = f["projectors"][spe][str(l)].shape[1]
        return cls(species, lmax, nmax, widths)

    def channel_rows(self, symbols):
        """Rows of every channel for one configuration, from its atomic symbols.
        
        Loop over the rows of psi in the order rkhs_vector.py writes them:
        atom -> l -> n -> m (this matches the order of the overlap matrix)
        starts[(spe, l)] collects, atom by atom for the atoms of species spe,
        an array with the first row of each of its (l, n) blocks
        """
        
        starts = {}
        i = 0
        for spe in symbols:  # one entry per atom of the configuration
            if spe not in self.spe_cols:
                raise LayoutMismatch(f"species {spe} is not in the species list")
            for l in range(self.lmax[spe] + 1):
                nn = self.nmax[(spe, l)]
                # the nn blocks of this atom + l follow each other
                starts.setdefault((spe, l), []).append(i + (2 * l + 1) * np.arange(nn))
                i += nn * (2 * l + 1)
        rows = {}
        for (spe, l), st in starts.items():
            st = np.array(st)  # (natoms_spe, nmax): first row of each (atom, n) block
            nat, nn = st.shape
            m = np.arange(2 * l + 1)
            # st.T[n, atom] + m gives shape (nmax, natoms_spe, 2l+1); flattening the last two
            # axes gives, in row n, the rows of channel (spe, l, n)
            rows[(spe, l)] = (st.T[:, :, np.newaxis] + m).reshape(nn, nat * (2 * l + 1))
        return i, rows # i: total number of rows, to be compared with psi.shape[0]


# Buffers kept between calls
_workspace = {}


def _flat(name, size):
    """Return a flat float64 buffer of at least size elements."""
    buf = _workspace.get(name)
    if buf is None or buf.size < size:
        _workspace.pop(name, None)
        buf = np.empty(size)
        _workspace[name] = buf
    return buf


def _view(name, shape):
    """Return a C-contiguous float64 array of the given shape, backed by a kept buffer."""
    size = shape[0] * shape[1]
    return _flat(name, size)[:size].reshape(shape)


def hessian_contribution(over, psivec, layout, symbols, bmat):
    """Add psivec^T @ over @ psivec to bmat (C-contiguous float64, shape (K, K))

    Returns the wall times in seconds of "step1" (T = over @ psivec) and "step2"
    (bmat += psivec^T @ T).
    """
    N_df, K = psivec.shape
    if bmat.shape != (K, K) or bmat.dtype != np.float64 or not bmat.flags.c_contiguous:
        raise ValueError(f"bmat must be a C-contiguous float64 array of shape ({K}, {K})")
    nrows, rows = layout.channel_rows(symbols)
    if (nrows, layout.ncols) != (N_df, K) or over.shape != (N_df, N_df):
        raise LayoutMismatch(
            f"expected psi of shape ({nrows}, {layout.ncols}) from the metadata, "
            f"found {psivec.shape} (overlap {over.shape})"
        )

    # tables P_{s,l}: one table per (species, l), zero-width tables are skipped
    psi = psivec.tocsr() # convert to CSR for efficient row slicing
    check = os.environ.get("SALTED_DENSE_BLOCKS_CHECK", "0") == "1"
    tables = {}
    nnz = 0
    for (spe, l), r in rows.items():
        M = layout.widths[(spe, l)] # number of columns of channel (spe, l, n)
        if M == 0 or r.shape[0] == 0: # skip empty channels
            continue
        c0 = layout.col0[(spe, l)] # first column of channel (spe, l, n=0)
        P = psi[r[0]][:, c0 : c0 + M].toarray() # shape (natoms_spe * (2l+1), M)
        tables[(spe, l)] = P # store P
        nnz += r.shape[0] * np.count_nonzero(P) # count non-zeros
        if check:
            for n in range(1, r.shape[0]):
                if not np.array_equal(psi[r[n]][:, c0 + n * M : c0 + (n + 1) * M].toarray(), P):
                    raise LayoutMismatch(f"block of channel ({spe}, l={l}, n={n}) differs from n=0")
    if nnz != psi.nnz:
        raise LayoutMismatch(f"psi has {psi.nnz} non-zeros, {nnz} of them inside the expected blocks")

    # columns of T: for the species present, in the order of the columns of psi
    tcols = {}  # spe -> (first column in T, first column in psi, number of columns)
    kt = 0
    for spe in layout.species:
        if any(key[0] == spe for key in tables): # species present
            a, b = layout.spe_cols[spe]
            tcols[spe] = (kt, a, b - a)
            kt += b - a # total number of columns of T for the species present
    if kt == 0:
        raise LayoutMismatch("no channel with non-zero width in this configuration")
    maxrows = max(rows[key].shape[1] for key in tables)
    gather = _flat("gather", maxrows * max(N_df, kt))
    t1 = time.time()

    # step 1: T[:, cols_c] = over[rows_c, :]^T @ P   (over is symmetric)
    T = _view("T", (N_df, kt)) # C-contiguous float64 array of shape (N_df, kt)
    for (spe, l), P in tables.items():
        r = rows[(spe, l)]
        nr, M = P.shape
        t_first, a, _ = tcols[spe]
        c0 = t_first + layout.col0[(spe, l)] - a
        for n in range(r.shape[0]): # loop over radial functions
            Sg = gather[: nr * N_df].reshape(nr, N_df)
            np.take(over, r[n], axis=0, out=Sg, mode="clip")
            np.matmul(Sg.T, P, out=T[:, c0 + n * M : c0 + (n + 1) * M]) # over[rows_c, :]^T @ P
    t2 = time.time()

    # step 2: bmat[cols_c, :] += P^T @ T[rows_c, :] (T only has columns for the species present)
    # If all species are present, the product is added into bmat directly.
    # Otherwise the product for each present species is computed separately and added to its block of bmat.
    for (spe, l), P in tables.items():
        r = rows[(spe, l)]
        nr, M = P.shape
        c0 = layout.col0[(spe, l)]
        for n in range(r.shape[0]):
            Tg = gather[: nr * kt].reshape(nr, kt)
            np.take(T, r[n], axis=0, out=Tg, mode="clip")
            out_rows = slice(c0 + n * M, c0 + (n + 1) * M)
            if kt == K: # all species present, add directly into bmat
                # bmat[cols_c, :] += P^T @ T[rows_c, :]
                target = bmat[out_rows, :].T
                res = dgemm(1.0, Tg.T, P.T, beta=1.0, c=target, trans_b=1, overwrite_c=1)
                if res.ctypes.data != target.ctypes.data:  # dgemm worked on a copy (future-proofing)
                    target[...] = res # copy back into bmat
            else: # some species missing, add into the columns of each present species
                for t_first, a, w in tcols.values(): # a: first column of the species in psi, w: number of columns of the species
                    bmat[out_rows, a : a + w] += P.T @ Tg[:, t_first : t_first + w]
    t3 = time.time()

    return {"step1": t2 - t1, "step2": t3 - t2}
