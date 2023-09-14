import numpy as np
cimport numpy as np
from sys import exit

cdef extern from "pyacofwi2D.h":
    void fwi_2D(char input_file[200], float *vel_inner, float *record_syn, float *record_obs,
                             float *grad, int verbose)
    void forward_2D(char input_file[200], float *vel_inner, float *record_syn, int verbose)


def fwi(np.ndarray[double, ndim=1, mode="c"] vel not None,
            np.ndarray[float, ndim=1, mode="c"] record_obs not None,
            paramfile = "./input/input_param.txt",
            verbose = 0):
    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()

    #np.savetxt('test.txt',vel)

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)
    # cdef np.ndarray[float, ndim=1,mode="c"] record_obs_f = np.zeros(record_obs.shape[0], dtype=np.float32)
    # record_obs_f = np.float32(record_obs)

    cdef np.ndarray[float, ndim=1, mode="c"] grad = np.zeros(vel_f.shape[0],dtype=np.float32)
    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(record_obs.shape[0],dtype=np.float32)

    fwi_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], &record_obs[0], &grad[0], verbose)

    return record_syn, grad


def forward(np.ndarray[double, ndim=1, mode="c"] vel not None,
            dim = 1, 
            paramfile = "./input/input_param.txt",
            verbose = 0):
    if(np.isnan(vel).any()):
        print('NaN occured in python')
        exit()

    cdef np.ndarray[float, ndim=1,mode="c"] vel_f = np.zeros(vel.shape[0], dtype=np.float32)
    vel_f = np.float32(vel)

    cdef np.ndarray[float, ndim=1, mode="c"] record_syn = np.zeros(dim, dtype=np.float32)

    forward_2D(str.encode(paramfile), &vel_f[0], &record_syn[0], verbose)

    return record_syn
