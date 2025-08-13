"""
SALTED OpenMP Sparse Matrix Operations

Minimal wrapper for OpenMP-accelerated dense @ sparse matrix multiplication
optimized for SALTED scientific computing workloads.

This module provides a simple interface to the compiled OpenMP Fortran functions
integrated into SALTED's build system.
"""

from typing import Union

import numpy as np
import scipy.sparse

from salted.lib import omp_sparse as _omp_sparse_lib


def dense_dot_sparse(
    dense_matrix: np.ndarray, sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix]
) -> np.ndarray:
    """
    Compute dense matrix matmul sparse_matrix using OpenMP acceleration.

    Args:
        dense_matrix: Dense matrix of shape (M, K) with dtype float64
        sparse_matrix: Sparse matrix of shape (K, N) in COO or CSC format

    Returns:
        Result matrix (= dense @ sparse) of shape (M, N) as numpy array
    """
    # Input validation
    if not isinstance(dense_matrix, np.ndarray):
        raise TypeError("Dense matrix must be a numpy array")
    if dense_matrix.ndim != 2:
        raise ValueError("Dense matrix must be 2-dimensional")
    if not scipy.sparse.issparse(sparse_matrix):
        raise TypeError("Sparse matrix must be a scipy sparse matrix")
    if sparse_matrix.ndim != 2:
        raise ValueError("Sparse matrix must be 2-dimensional")
    if dense_matrix.shape[1] != sparse_matrix.shape[0]:
        raise ValueError(
            f"Matrix dimension mismatch: dense matrix has {dense_matrix.shape[1]} columns, "
            f"sparse matrix has {sparse_matrix.shape[0]} rows"
        )

    # Ensure proper data types and memory layout
    dense_matrix = np.ascontiguousarray(dense_matrix, dtype=np.float64)

    # Convert to CSC format if needed
    if not isinstance(sparse_matrix, scipy.sparse.csc_matrix):
        sparse_matrix = sparse_matrix.tocsc()

    # Extract CSC components with correct data types
    csc_indptr = np.ascontiguousarray(sparse_matrix.indptr, dtype=np.int32)
    csc_indices = np.ascontiguousarray(sparse_matrix.indices, dtype=np.int32)
    csc_data = np.ascontiguousarray(sparse_matrix.data, dtype=np.float64)

    # Matrix dimensions
    M, K = dense_matrix.shape
    N = sparse_matrix.shape[1]

    # Call OpenMP Fortran function
    return _omp_sparse_lib.omp_sparse_mod.dense_dot_sparse(dense_matrix, csc_indptr, csc_indices, csc_data, M, K, N)


def sparse_transpose_dot_dense(
    sparse_matrix: Union[scipy.sparse.coo_matrix, scipy.sparse.csc_matrix], dense_matrix: np.ndarray
) -> np.ndarray:
    """
    Compute sparse_matrix.T matmul dense matrix or dense vector using OpenMP acceleration.

    Args:
        sparse_matrix: Sparse matrix of shape (M, N)
        dense_matrix: Dense matrix of shape (M, K) or dense vector of shape (M,)

    Returns:
        Result vector/matrix (= sparse.T @ dense) of shape (N,) or (N, K)
    """
    # Handle vector input by reshaping to matrix
    if dense_matrix.ndim == 1:
        dense_matrix = dense_matrix.reshape(-1, 1)
        squeeze_result = True
    else:
        squeeze_result = False

    # Transpose operation: (M,N).T @ (M,K) = (N,K)
    # This is equivalent to: (K,M) @ (M,N) = (K,N), then transpose to (N,K)
    result = dense_dot_sparse(
        dense_matrix.T,  # (K,M)
        sparse_matrix,  # (M,N)
    ).T  # (N,K)

    return result.squeeze() if squeeze_result else result


def _test_wrapper():
    """Test the OpenMP sparse wrapper with SALTED-style operations."""
    import time

    print("Testing SALTED OpenMP Sparse Wrapper")

    # Create test matrices similar to SALTED usage
    nfeatures, nsamples, nvecs = 1000, 800, 500

    # Create sparse psi matrix
    psi_sparse = scipy.sparse.random(nfeatures, nsamples, density=0.02, format="coo")

    # Create dense vectors (typical SALTED reference data)
    ref_projs = np.random.randn(nfeatures).astype(np.float64)
    dense_matrix = np.random.randn(nvecs, nfeatures).astype(np.float64)  # (50, 1000)

    print(f"\nTest matrices:")
    print(
        f"  psi_sparse: {psi_sparse.shape} ({psi_sparse.nnz} nnz, {psi_sparse.nnz / (nfeatures * nsamples):.1%} density)"
    )
    print(f"  ref_projs: {ref_projs.shape}")
    print(f"  dense_matrix: {dense_matrix.shape}")

    print("\n=== Test 1: dense_dot_sparse ===")

    # Test dense @ sparse (standard usage)
    # dense_matrix is (50, 1000), psi_sparse is (1000, 800)
    print("Testing: dense_matrix @ psi_sparse")
    start_time = time.time()
    result_omp = dense_dot_sparse(dense_matrix, psi_sparse)
    omp_time = time.time() - start_time

    start_time = time.time()
    result_numpy = dense_matrix @ psi_sparse.toarray()
    numpy_time = time.time() - start_time

    max_diff = np.max(np.abs(result_omp - result_numpy))
    rel_error = max_diff / (np.max(np.abs(result_numpy)) + 1e-15)

    print(f"Result shape: {result_omp.shape}")
    print(f"OpenMP time: {omp_time:.4f}s, NumPy time: {numpy_time:.4f}s, Speedup: {numpy_time / omp_time:.2f}x")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Relative error: {rel_error:.2e}")
    print(f"Test {'PASSED' if rel_error < 1e-12 else 'FAILED'}")

    print("\n=== Test 2: sparse_transpose_dot_dense (SALTED pattern) ===")

    # Test psi.T @ ref_projs (typical SALTED operation)
    print("Testing: psi_sparse.T @ ref_projs")
    start_time = time.time()
    result_omp = sparse_transpose_dot_dense(psi_sparse, ref_projs)
    omp_time = time.time() - start_time

    start_time = time.time()
    result_numpy = psi_sparse.T.toarray() @ ref_projs
    numpy_time = time.time() - start_time

    max_diff = np.max(np.abs(result_omp - result_numpy))
    rel_error = max_diff / (np.max(np.abs(result_numpy)) + 1e-15)

    print(f"Result shape: {result_omp.shape}")
    print(f"OpenMP time: {omp_time:.4f}s, NumPy time: {numpy_time:.4f}s, Speedup: {numpy_time / omp_time:.2f}x")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Relative error: {rel_error:.2e}")
    print(f"Test {'PASSED' if rel_error < 1e-12 else 'FAILED'}")

    print("\n=== Test 3: sparse_transpose_dot_dense (matrix) ===")

    # Test psi.T @ dense_matrix.T (matrix version)
    # psi_sparse.T is (800, 1000), dense_matrix.T is (1000, 50)
    print("Testing: psi_sparse.T @ dense_matrix.T")
    start_time = time.time()
    result_omp = sparse_transpose_dot_dense(psi_sparse, dense_matrix.T)
    omp_time = time.time() - start_time

    start_time = time.time()
    result_numpy = psi_sparse.T.toarray() @ dense_matrix.T
    numpy_time = time.time() - start_time

    max_diff = np.max(np.abs(result_omp - result_numpy))
    rel_error = max_diff / (np.max(np.abs(result_numpy)) + 1e-15)

    print(f"Result shape: {result_omp.shape}")
    print(f"OpenMP time: {omp_time:.4f}s, NumPy time: {numpy_time:.4f}s, Speedup: {numpy_time / omp_time:.2f}x")
    print(f"Max difference: {max_diff:.2e}")
    print(f"Relative error: {rel_error:.2e}")
    print(f"Test {'PASSED' if rel_error < 1e-12 else 'FAILED'}")

    print("\nAll wrapper tests completed!")
    return True


# Export main interface
__all__ = [
    "dense_dot_sparse",
    "sparse_transpose_dot_dense",
]


if __name__ == "__main__":
    _test_wrapper()
