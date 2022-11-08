import numpy as np
from vip.tomo.pyfm2d import fm2d_lglike

def run_tomo(vel, data, src, rec, config, client=None):
    '''
    Call tomography code to get loss and gradient
    Input
        vel: velocity of all particles, shape (n, nparameters)
        data: a vector of travel time data
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


