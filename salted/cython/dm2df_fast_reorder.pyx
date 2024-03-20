import cython
cimport numpy as cnp
cnp.import_array()

ctypedef cnp.float64_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def reorder(cnp.ndarray[DTYPE_t, ndim=1] rho, cnp.ndarray[DTYPE_t, ndim=2] overlap, object symb, dict lmax, dict nmax):
    cdef cnp.ndarray[DTYPE_t, ndim=1] Coef = rho.copy()
    cdef cnp.ndarray[DTYPE_t, ndim=2] Over = overlap.copy()
    cdef int i1 = 0, l1, n1, im1, i2, l2, n2, im2
    cdef object spe1, spe2
    
    for spe1 in symb:
        for l1 in range(lmax[spe1]+1):
            for n1 in range(nmax[(spe1, l1)]):
                for im1 in range(2 * l1 + 1):
                    if l1 == 1 and im1 == 2:
                        Coef[i1] = rho[i1 - 2]
                    elif l1 == 1 and im1 != 2:
                        Coef[i1] = rho[i1 + 1]
                    i2 = 0
                    for spe2 in symb:
                        for l2 in range(lmax[spe2]+1):
                            for n2 in range(nmax[(spe2,l2)]):
                                for im2 in range(2*l2+1):
                                    if l1 == 1 and l2 != 1:
                                        if im1 != 2:
                                            Over[i1, i2] = overlap[i1 + 1, i2]
                                        else:
                                            Over[i1, i2] = overlap[i1 - 2, i2]
                                    elif l1 != 1 and l2 == 1:
                                        if im2 != 2:
                                            Over[i1, i2] = overlap[i1, i2 + 1]
                                        else:
                                            Over[i1, i2] = overlap[i1, i2 - 2]
                                    elif l1 == 1 and l2 == 1:
                                        if im1 != 2 and im2 != 2:
                                            Over[i1, i2] = overlap[i1 + 1, i2 + 1]
                                        elif im1 != 2 and im2 == 2:
                                            Over[i1, i2] = overlap[i1 + 1, i2 - 2]
                                        elif im1 == 2 and im2 != 2:
                                            Over[i1, i2] = overlap[i1 - 2, i2 + 1]
                                        else:
                                            Over[i1, i2] = overlap[i1 - 2, i2 - 2]
                                    i2 += 1
                    i1 += 1
    return Coef, Over
