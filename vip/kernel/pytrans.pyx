import numpy as np
cimport numpy as np

cdef extern from "pytrans.h":
    void transform_c(int *, double *, double *, double *, double *);
    void inv_transform_c(int *, double *, double *, double *, double *);
    void log_jacobian_c(int *, double *, double *, double *, double *);
    void trans_grad_c(int *, double *, double *, double *, double *);
    void many_transform_c(int *, int *, double *, double *, double *);
    void many_inv_transform_c(int *, int *, double *, double *, double *);
    void many_log_jacobian_c(int *, int *, double *, double *, double *, double *);
    void many_trans_grad_c(int *, int *, double *, double *, double *, double *);
    void many_trans_grad_c2(int *, int *, double *, double *, double *, double *, int *);

def pytrans(np.ndarray[double,ndim=2] x,
            np.ndarray[double,ndim=1] lb,
            np.ndarray[double,ndim=1] ub):

    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]
    many_transform_c(&m,&n,&x[0,0],&lb[0],&ub[0])
    
    return x

def pyinv_trans(np.ndarray[double,ndim=2] x,
            np.ndarray[double,ndim=1] lb,
            np.ndarray[double,ndim=1] ub):

    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]
    many_inv_transform_c(&m,&n,&x[0,0],&lb[0],&ub[0])
    
    return x

def pyjac(np.ndarray[double,ndim=2] x,
            np.ndarray[double,ndim=1] lb,
            np.ndarray[double,ndim=1] ub):

    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]

    cdef np.ndarray[double,ndim=1] jac = np.zeros((m,), dtype=np.float64)
    many_log_jacobian_c(&m,&n,&x[0,0],&lb[0],&ub[0],&jac[0])
    return jac

def pytrans_grad(np.ndarray[double,ndim=2] grad,
                 np.ndarray[double,ndim=2] x,
            np.ndarray[double,ndim=1] lb,
            np.ndarray[double,ndim=1] ub,
            int rmask=0):

    cdef int m, n
    m = x.shape[0]
    n = x.shape[1]
    cdef np.ndarray[int,ndim=2] mask = np.zeros((m,n), dtype=np.int32)
    if(rmask):
        many_trans_grad_c2(&m,&n,&grad[0,0],&x[0,0],&lb[0],&ub[0],&mask[0,0])
    else:
        many_trans_grad_c(&m,&n,&grad[0,0],&x[0,0],&lb[0],&ub[0])
    return grad, np.array(mask,dtype=bool)

