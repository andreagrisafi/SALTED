module omp_sparse_mod

    use omp_lib
    implicit none

contains

subroutine dense_dot_sparse(dense_mat, csc_indptr, csc_indices, csc_data, m, k, n, result)
    use omp_lib
    implicit none

    !f2py intent(in) :: m, k, n
    !f2py depend(m, k) :: dense_mat
    !f2py intent(in) :: csc_indptr, csc_indices, csc_data
    !f2py depend(n) :: csc_indptr
    !f2py intent(out) :: result
    !f2py depend(m, n) :: result

    integer, intent(in) :: m, k, n
    real(kind=8), dimension(m, k), intent(in) :: dense_mat
    integer, dimension(n + 1), intent(in) :: csc_indptr
    integer, dimension(:), intent(in) :: csc_indices
    real(kind=8), dimension(:), intent(in) :: csc_data
    real(kind=8), dimension(m, n), intent(out) :: result

    integer :: j, l, row_idx, nnz
    real(kind=8) :: sparse_val

    nnz = size(csc_indices)
    result = 0.0d0

    !$omp parallel default(private) shared(result, dense_mat, csc_indptr, csc_indices, csc_data, m, k, n)

    !$omp single
    print *, "omp_sparse: Using", omp_get_num_threads(), "threads in OpenMP"
    !$omp end single

    !$omp do schedule(dynamic, 16)
    do j = 1, n ! Loop over columns of sparse matrix
        do l = csc_indptr(j) + 1, csc_indptr(j + 1)
            row_idx = csc_indices(l) + 1  ! Convert 0-based to 1-based
            sparse_val = csc_data(l)

            ! Vectorized operation for better performance
            result(:, j) = result(:, j) + dense_mat(:, row_idx) * sparse_val
        end do
    end do
    !$omp end do

    !$omp end parallel

end subroutine

end module omp_sparse_mod
