import numpy as np
import sys
sys.path.insert(0,'/otherdata/zhxls7/variational/svgd')
from pytrans import *

def batch_trans(x,batchsize=20,trans=1,lb=0,ub=1e8):
    npar = x.shape[0]
    for i in range(0,npar,batchsize):
        end = min(i+batchsize,npar)
        if(trans==1):
            x[i:end,:] = pytrans(np.ascontiguousarray(x[i:end,:]),lb,ub)
        else:
            x[i:end,:] = pyinv_trans(np.ascontiguousarray(x[i:end,:]),lb,ub)
    return x

def batch_trans_grad(grad,x,mask,batchsize=20,lb=0,ub=1e8):
    npar = x.shape[0]
    for i in range(0,npar,batchsize):
        end = min(i+batchsize,npar)
        g, m = pytrans_grad(np.ascontiguousarray(grad[i:end,:]),np.ascontiguousarray(x[i:end,:]),lb,ub,rmask=1)
        grad[i:end,:] = g
        mask[i:end,:] = m
    return grad, mask
