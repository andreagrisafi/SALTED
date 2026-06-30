"""
Numba JIT kernel for dense @ CSC sparse matrix multiplication, used to
accelerate the Hessian H = Psi^T @ S @ Psi in hessian_matrix.py.

Only the K-blocked variant is exposed here; it is strictly better than the
plain column-parallel kernel for the matrix sizes encountered in SALTED
(N_df up to ~17000, where AT = S.T can reach several GB).

Public API:
    get_hessian_engine(N_df, K_rkhs) -> KBlockedHessianEngine
        Returns a cached engine for the given shape.
        Cache budget is read from SALTED_NUMBA_CACHE_BYTES (default 32 MiB).
"""

import os
import re
import subprocess

import numpy as np
import scipy.sparse
from numba import njit, prange


# ---------------------------------------------------------------------------
# Cache-size detection
# ---------------------------------------------------------------------------


def _detect_cache_bytes():
    """L3 byte budget for the AT k-block.

    Resolution order:
      1. SALTED_NUMBA_CACHE_BYTES env var  (use for MPI: set per-rank budget)
      2. lscpu L3 / 2
      3. 32 MiB fallback
    """
    env_val = os.environ.get("SALTED_NUMBA_CACHE_BYTES")
    if env_val:
        return int(env_val)
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        m = re.search(r"L3 cache:\s+([\d.]+)\s*([KMG])i?B", out, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            unit = m.group(2).upper()
            l3 = int(val * {"K": 1024, "M": 1024**2, "G": 1024**3}[unit])
            return l3 // 2
    except Exception:
        pass
    return 32 * 1024 * 1024


def _auto_bk(M, cache_bytes, itemsize=8):
    """Rows of AT per k-block given a byte budget."""
    return max(64, cache_bytes // max(1, int(M) * itemsize))


# ---------------------------------------------------------------------------
# K-blocked column-parallel kernel
# ---------------------------------------------------------------------------


@njit(parallel=True, fastmath=True, cache=True)
def _dense_csc_kernel_kblocked(dense_T, indptr, indices, data, result_T, B_k):
    # dense_T : (K, M) C-order
    # result_T: (N, M) C-order, pre-zeroed by caller
    N, M = result_T.shape
    K = dense_T.shape[0]
    n_kblocks = (K + B_k - 1) // B_k
    for kb in range(n_kblocks):  # serial
        k0 = kb * B_k
        k1 = min(k0 + B_k, K)
        for j in prange(N):  # parallel over columns
            col_lo = indptr[j]
            col_hi = indptr[j + 1]
            lo, hi = col_lo, col_hi
            # below is optimized, using bisect search
            while lo < hi:
                mid = (lo + hi) >> 1
                if indices[mid] < k0:
                    lo = mid + 1
                else:
                    hi = mid
            ptr_lo = lo
            lo, hi = ptr_lo, col_hi
            while lo < hi:
                mid = (lo + hi) >> 1
                if indices[mid] < k1:
                    lo = mid + 1
                else:
                    hi = mid
            ptr_hi = lo
            for ptr in range(ptr_lo, ptr_hi):
                k = indices[ptr]
                val = data[ptr]
                for i in range(M):
                    result_T[j, i] += dense_T[k, i] * val


# ---------------------------------------------------------------------------
# Engine: pre-allocated buffers + two-step Hessian
# ---------------------------------------------------------------------------


class KBlockedHessianEngine:
    """
    H = Psi^T @ S @ Psi via two K-blocked SpMM calls.

    Pre-allocates all buffers for the given (N_df, K_rkhs) shape.
    Call warmup() once before the first timed/MPI use to trigger JIT compilation.
    """

    def __init__(self, N_df, K_rkhs, cache_bytes=None):
        cb = cache_bytes if cache_bytes is not None else _detect_cache_bytes()
        itemsize = 8  # float64
        self.R_storage = np.empty((K_rkhs, N_df), dtype=np.float64)
        self._R_buf = np.empty((N_df, K_rkhs), dtype=np.float64)
        self.H = np.empty((K_rkhs, K_rkhs), dtype=np.float64)
        self._B_k1 = int(_auto_bk(N_df, cb, itemsize))  # step 1: rows of S
        self._B_k2 = int(_auto_bk(K_rkhs, cb, itemsize))  # step 2: rows of R

    def compute(self, S, Psi_csc):
        """Return H = Psi^T @ S @ Psi, shape [K_rkhs, K_rkhs].

        Returns a view into an internal buffer; call .copy() if you need to own it.
        """
        if not scipy.sparse.issparse(Psi_csc):
            raise TypeError("Psi_csc must be a scipy sparse matrix")
        sp = Psi_csc if isinstance(Psi_csc, scipy.sparse.csc_matrix) else Psi_csc.tocsc()
        indptr = np.ascontiguousarray(sp.indptr, dtype=np.int32)
        indices = np.ascontiguousarray(sp.indices, dtype=np.int32)
        data = np.ascontiguousarray(sp.data, dtype=np.float64)
        S_c = np.ascontiguousarray(S, dtype=np.float64)

        # Step 1: R^T = Psi^T @ S (S is symmetric → pass S directly as dense_T)
        self.R_storage[:] = 0.0
        _dense_csc_kernel_kblocked(S_c, indptr, indices, data, self.R_storage, self._B_k1)

        # Step 2: H = Psi^T @ R (R_storage.T is strided → copy required)
        np.copyto(self._R_buf, self.R_storage.T)
        self.H[:] = 0.0
        _dense_csc_kernel_kblocked(self._R_buf, indptr, indices, data, self.H, self._B_k2)
        return self.H  # H is symmetric

    @staticmethod
    def warmup():
        """Trigger JIT compilation with a zero-work dummy call."""
        iptr = np.zeros(2, np.int32)
        idx = np.zeros(1, np.int32)
        dat = np.zeros(1, np.float64)
        buf = np.zeros((1, 1), np.float64)
        _dense_csc_kernel_kblocked(np.zeros((1, 1)), iptr, idx, dat, buf, 1)


# ---------------------------------------------------------------------------
# Module-level engine cache
# ---------------------------------------------------------------------------

_engines: dict = {}


def get_hessian_engine(N_df, K_rkhs):
    """Return a cached KBlockedHessianEngine for the given shape.

    JIT-compiles on first call (once per process).
    Cache budget read from SALTED_NUMBA_CACHE_BYTES env var (default 32 MiB).
    For MPI jobs set: export SALTED_NUMBA_CACHE_BYTES=$(( L3_bytes / ranks_per_node / 2 ))
    """
    key = (N_df, K_rkhs)
    if key not in _engines:
        cache_bytes = _detect_cache_bytes()
        eng = KBlockedHessianEngine(N_df, K_rkhs, cache_bytes=cache_bytes)
        KBlockedHessianEngine.warmup()
        _engines[key] = eng
    return _engines[key]
