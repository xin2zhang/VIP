import numpy as np
from forward.tomo2d.pyfm2d import fm2d_lglike, many_fm2d

def run_tomo(vel, data, src, rec, config, client=None):
    '''
    Call tomography code to get loss and gradient
    Input
        vel: velocity of all particles, shape (n, nparameters)
        data: a 2d array of travel time data, shape (nsrc*nrec,2),
              1st column is travel time, 2nd column is noise
              set travel time to zero if no data is available
        src, rec: array of source/receiver locations, shape (n, 2)
        config: configure.ConfigureParser()
        client: a dask client, not used here
    '''

    ns = src.shape[0]
    nr = rec.shape[0]
    status = np.zeros((2,ns*nr),dtype=np.int32)
    w = data[:,0] > 0
    status[0,w] = 1
    status[1,:] = np.linspace(1,ns*nr,ns*nr)

    data[data[:,1]<=0, 1] = 1.0 # for safety, set the noise to nonzero if it is zero

    srcx = np.ascontiguousarray(src[:,0])
    srcy = np.ascontiguousarray(src[:,1])
    recx = np.ascontiguousarray(rec[:,0])
    recy = np.ascontiguousarray(rec[:,1])

    nx = config.getint('tomo','nx')
    ny = config.getint('tomo','ny')
    xmin = config.getfloat('tomo','xmin')
    ymin = config.getfloat('tomo','ymin')
    dx = config.getfloat('tomo','dx')
    dy = config.getfloat('tomo','dy')
    gdx = config.getint('tomo','gdx')
    gdy = config.getint('tomo','gdy')
    sdx = config.getint('tomo','sdx')
    sext = config.getint('tomo','sext')
    earth = config.getfloat('tomo','earth')

    res, grads = fm2d_lglike(vel, srcx, srcy, recx, recy,
                             status, nx, ny, xmin, ymin,
                             dx, dy, gdx, gdy, sdx, sext,
                             data, earth)

    return res, grads

def run_many_tomo(vel, data, src, rec, config, client=None):
    '''
    Call tomography code to get loss and gradient
    Input
        vel: velocity of all models at different periods, shape (np, nparameters)
        data: a 3d array of travel time data, shape (np,nsrc*nrec,2)
              data at all np periods, details see above
        src, rec: array of source/receiver locations, shape (n, 2)
        config: configure.ConfigureParser()
        client: a dask client, not used here
    '''

    ns = src.shape[0]
    nr = rec.shape[0]
    srcx = np.ascontiguousarray(src[:,0])
    srcy = np.ascontiguousarray(src[:,1])
    recx = np.ascontiguousarray(rec[:,0])
    recy = np.ascontiguousarray(rec[:,1])

    data[data[:,:,1]<=0, 1] = 1.0 # for safety, set the noise to nonzero if it is zero

    nx = config.getint('tomo','nx')
    ny = config.getint('tomo','ny')
    xmin = config.getfloat('tomo','xmin')
    ymin = config.getfloat('tomo','ymin')
    dx = config.getfloat('tomo','dx')
    dy = config.getfloat('tomo','dy')
    gdx = config.getint('tomo','gdx')
    gdy = config.getint('tomo','gdy')
    sdx = config.getint('tomo','sdx')
    sext = config.getint('tomo','sext')
    earth = config.getfloat('tomo','earth')

    res, grads = many_fm2d(vel, srcx, srcy, recx, recy,
                             nx, ny, xmin, ymin,
                             dx, dy, gdx, gdy, sdx, sext,
                             data, earth)

    return res, grads


