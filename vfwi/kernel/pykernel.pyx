import numpy as np
cimport numpy as np

cdef extern from "pykernel.h":
    void sqrd_c(int, int, double *, double *, double *);
    void svgd_grad(int, int, double *, double *, int, double *, double *, double, double *);
    void ksd(int, int, double *, double *, int, double *, double *, double, double *);

def pdist(np.ndarray[double,ndim=2] x,
          np.ndarray[double,ndim=1] w):
    '''
    Compute pair-wise distance
    Input
        x: array of variables, shape (n, ndim)
        w: weight for calculating distance, a vector of (ndim,)
    Return
        dist: pair-wise distance, shape (n, n)
    '''
    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]
    cdef np.ndarray[double,ndim=2] dist = np.zeros((m,m),dtype=np.float64)
    sqrd_c(m,n,&x[0,0],&w[0],&dist[0,0])

    return dist

def svgd_gradient(np.ndarray[double,ndim=2] x,
              np.ndarray[double,ndim=2] grad,
              np.ndarray[double,ndim=2] dist,
              np.ndarray[double,ndim=1] w,
              str kernel='diagonal',double h=1):
    '''
    Compute svgd gradient
    Input
        x: array of variables, shape (n, nidm)
        grad: gradient of posterior w.r.t x for each partilce, shape (n, ndim)
        dist: pair-wise distance, shape (n, n)
        w: weight for kernel calculation, i.e. a diagonal matrix kernel, shape (ndim, )
        kernel: kernel function, rbf or diagonal
        h: width for rbf kernel, default to median trick
    Return
        grad_out: svgd gradients
    '''
    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]

    medh = np.median(dist)
    medh = np.sqrt(0.5*medh/np.log(m+1))
    if(h<0):
        h = medh
    else:
        h = h*medh

    cdef np.ndarray[double,ndim=2] grad_out = np.zeros((m,n),dtype=np.float64)
    cdef int ikernel = 0
    if(kernel=='rbf'):
        ikernel = 1 
    elif(kernel=='diagonal'):
        ikernel = 2 
    svgd_grad(m,n,&x[0,0],&grad[0,0],ikernel,&dist[0,0],&w[0],h,&grad_out[0,0])

    return grad_out

def pyksd(np.ndarray[double,ndim=2] x,
              np.ndarray[double,ndim=2] grad,
              np.ndarray[double,ndim=1] w,
              str kernel='diagonal',double h=-1):
    '''
    Compute kernelized Stein discrepancy
    Input
        x: array of variables, shape (n, nidm)
        grad: gradient of posterior w.r.t x for each partilce, shape (n, ndim)
        w: weight for kernel calculation, i.e. a diagonal matrix kernel, shape (ndim, )
        kernel: kernel function, rbf or diagonal
        h: width for rbf kernel, default to median trick
    Return
        ksd_value: kernelzied Stein discrepancy
        stepsize: the upper bound of the stepsize estimated using ksd value
    '''
    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]

    cdef np.ndarray[double,ndim=2] dist = np.zeros((m,m),dtype=np.float64)
    sqrd_c(m,n,&x[0,0],&w[0],&dist[0,0])

    medh = np.median(dist)
    medh = np.sqrt(0.5*medh/np.log(m+1))
    if(h<0):
        h = medh
    else:
        h = h*medh

    cdef double ksd_value = 0
    cdef int ikernel = 0
    if(kernel=='rbf'):
        ikernel = 1 
    elif(kernel=='diagonal'):
        ikernel = 2 
    ksd(m,n,&x[0,0],&grad[0,0],ikernel,&dist[0,0],&w[0],h,&ksd_value)
    stepsize = h/(4*np.sqrt(n)*ksd_value)

    return ksd_value, stepsize
