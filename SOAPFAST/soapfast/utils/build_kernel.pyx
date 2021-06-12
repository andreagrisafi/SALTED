cimport cython # the main Cython stuff

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cython.parallel cimport prange

cimport scipy.special.cython_special as csc # Cython interfaces to Scipy Special functions (sph_harm, gamma, hyp1f1)

from libc.math cimport sqrt,abs
from libc.stdio cimport printf

cdef extern void zero_array (double *array, double value, size_t n)

# NumPy and Scipy Special Functions
import numpy as np
import scipy.special as sc

import itertools

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _nonlinearscalarkernelf(
                    long npoints1,
                    long npoints2,
                    long[:] scale1,
                    long[:] scale2,
                    long featsize,
                    long zeta,
                    double[:,:,:] ps1,
                    double[:,:,:] ps2,
                    long use_hermiticity,
                    double[:,:] kreal):
    """
    compute kernel for lambda=0 and zeta>0 using prange on outer loop 
    """
    # Py_ssize_t is the correct C type for Python array indexes
    cdef Py_ssize_t i,j,iat,jat,k,lbound
    cdef double a,krn

    for i in prange(npoints1, nogil=True, schedule=dynamic):
        if (use_hermiticity==1):
            lbound = i
        else:
            lbound = 0
        for j in xrange(lbound,npoints2):
            for iat in xrange(scale1[i]):
                for jat in xrange(scale2[j]): 
                    krn = 0.0
                    for k in xrange(featsize):
                        krn = krn + ps1[i,iat,k]*(ps2[j,jat,k])
                    kreal[i,j] = kreal[i,j] + (krn)**zeta 
            kreal[i,j] = kreal[i,j]/(scale1[i]*scale2[j])

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _lineartensorkernelf(
                    long lam,
                    long npoints1,
                    long npoints2,
                    long featsize,
                    double[:,:,:] ps1,
                    double[:,:,:] ps2,
                    long use_hermiticity,
                    double[:,:,:,:] kernel):
    """
    compute kernel for lambda>0 and zeta>0 using prange on outer loop 
    """
    # Py_ssize_t is the correct C type for Python array indexes
    cdef Py_ssize_t i,j,k,mu,nu,lbound
    cdef double krn

    for i in prange(npoints1, nogil=True, schedule=dynamic):
        if (use_hermiticity==1):
            lbound = i
        else:
            lbound = 0
        for j in xrange(lbound,npoints2):
            for mu in xrange(2*lam+1): 
                for nu in xrange(2*lam+1): 
                    krn = 0.0
                    for k in xrange(featsize):
                        krn = krn + ps1[i,mu,k]*(ps2[j,nu,k])
                    kernel[i,j,mu,nu] =  kernel[i,j,mu,nu] + krn

##########################################################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _nonlineartensorkernelf(
                    long lam,
                    long npoints1,
                    long npoints2,
                    long[:] scale1,
                    long[:] scale2,
                    long featsize,
                    long featsize0,
                    long zeta,
                    double[:,:,:,:] ps1,
                    double[:,:,:,:] ps2,
                    double[:,:,:] ps01,
                    double[:,:,:] ps02,
                    long use_hermiticity,
                    double[:,:,:,:] kernel):
    """
    compute kernel for lambda>0 and zeta>1 using prange on outer loop 
    """
    # Py_ssize_t is the correct C type for Python array indexes
    cdef Py_ssize_t i,j,iat,jat,k,mu,nu,lbound
    cdef double krn,krn0

    for i in prange(npoints1, nogil=True, schedule=dynamic):
        if (use_hermiticity==1):
            lbound = i
        else:
            lbound = 0
        for j in xrange(lbound,npoints2):
            for iat in xrange(scale1[i]):
                for jat in xrange(scale2[j]):
                    krn0 = 0.0
                    for k in xrange(featsize0):
                         krn0 = krn0 + ps01[i,iat,k]*(ps02[j,jat,k])
                    for mu in xrange(2*lam+1): 
                        for nu in xrange(2*lam+1): 
                            krn = 0.0
                            for k in xrange(featsize):
                                krn = krn + ps1[i,iat,mu,k]*(ps2[j,jat,nu,k])
                            kernel[i,j,mu,nu] =  kernel[i,j,mu,nu] + krn * krn0**(zeta-1)

##########################################################################################################
##########################################################################################################

def calc_nonlinear_scalar_kernel(npoints,scale,featsize,zeta,PS,use_hermiticity):
    # Allocate output array
    kreal = np.zeros((npoints[0],npoints[1]),dtype=float)
    if (use_hermiticity):
        int_hermiticity = 1
    else:
        int_hermiticity = 0
    _nonlinearscalarkernelf(npoints[0],npoints[1],scale[0],scale[1],featsize,zeta,PS[0],PS[1],int_hermiticity,kreal) 
    return kreal 

##########################################################################################################

def calc_linear_tensor_kernel(lam,npoints,featsize,PS,use_hermiticity):
    if (use_hermiticity):
        int_hermiticity = 1
    else:
        int_hermiticity = 0
    kreal = np.zeros((npoints[0],npoints[1],2*lam+1,2*lam+1),dtype=float)
    _lineartensorkernelf(lam,npoints[0],npoints[1],featsize,PS[0],PS[1],int_hermiticity,kreal)
    return kreal

##########################################################################################################

def calc_nonlinear_tensor_kernel(lam,npoints,scale,featsize,featsize0,zeta,PS,PS0,use_hermiticity):
    if (use_hermiticity):
        int_hermiticity = 1
    else:
        int_hermiticity = 0
    kreal = np.zeros((npoints[0],npoints[1],2*lam+1,2*lam+1),dtype=float)
    _nonlineartensorkernelf(lam,npoints[0],npoints[1],scale[0],scale[1],featsize,featsize0,zeta,PS[0],PS[1],PS0[0],PS0[1],int_hermiticity,kreal)
    return kreal 
