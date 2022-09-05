import numpy as np
from vip.kernel.pytrans import *

def trans(x, batchsize=None, trans=1, lb=0, ub=1e8):
    '''
    Transform between constrained and unconstrained variable
    Input
        x: original variable, has a dimension (n,dim) where n is the number of variables
        batchsize: minibatch of variables, default to n
        trans: transform direction, 1 transform constrained to unconstrained, 0 inverse
        lb, ub: lower and upper bound, a vector of length dim
    Return
        x: transformed variable
    '''
    n = x.shape[0]
    if(batchsize is None):
        batchsize = n

    for i in range(0,n,batchsize):
        end = min(i+batchsize,n)
        if(trans==1):
            x[i:end,:] = pytrans(np.ascontiguousarray(x[i:end,:],dtype=np.float64),
                                 np.ascontiguousarray(lb,dtype=np.float64),
                                 np.ascontiguousarray(ub,dtype=np.float64))
        else:
            x[i:end,:] = pyinv_trans(np.ascontiguousarray(x[i:end,:],dtype=np.float64),
                                     np.ascontiguousarray(lb,dtype=np.float64),
                                     np.ascontiguousarray(ub,dtype=np.float64))
    return x

def trans_grad(grad, x, mask=None, batchsize=None, lb=0, ub=1e8):
    '''
    Gradient after transforming from constrained variable to unconstrained variable
    Input
        grad: gradient of original varialbe
        x:  transformed variable
        mask: an array to store the mask array
        batchsize: minibatch of variables, default to n
        lb, ub: lower and upper bound, a vector of length dim
    Return
        grad: transformed gradient
        mask: mask array
    '''
    n = x.shape[0]
    if(batchsize is None):
        batchsize = n

    for i in range(0,n,batchsize):
        end = min(i+batchsize,n)
        g, m = pytrans_grad(np.ascontiguousarray(grad[i:end,:],dtype=np.float64),
                         np.ascontiguousarray(x[i:end,:],dtype=np.float64),
                         np.ascontiguousarray(lb,dtype=np.float64),
                         np.ascontiguousarray(ub,dtype=np.float64),rmask=1)
        grad[i:end,:] = g
        if(mask is not None): mask[i:end,:] = m

    return grad, mask

