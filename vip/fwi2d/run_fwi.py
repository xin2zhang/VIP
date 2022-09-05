import numpy as np
from vip.fwi2d.aco2d import fwi

def prepare_config(config):

    paramfile = config.get('FWI','configfile')
    with open(paramfile,'w') as f:
        f.write('--grid points in x direction (nx)\n')
        f.write(config.get('FWI','nx')+'\n')
        f.write('--grid points in z direction (nz)\n')
        f.write(config.get('FWI','nz')+'\n')
        f.write('--pml points (pml0)\n')
        f.write(config.get('FWI','pml')+'\n')
        f.write('--Finite difference order (Lc)\n')
        f.write(config.get('FWI','Lc')+'\n')
        f.write('--Total number of sources (ns)\n')
        f.write(config.get('FWI','ns')+'\n')
        f.write('--Total time steps (nt)\n')
        f.write(config.get('FWI','nt')+'\n')
        f.write('--Shot interval in grid points (ds)\n')
        f.write(config.get('FWI','ds')+'\n')
        f.write('--Grid number of the first shot to the left of the model (ns0)\n')
        f.write(config.get('FWI','ns0')+'\n')
        f.write('--Depth of source in grid points (depths)\n')
        f.write(config.get('FWI','depths')+'\n')
        f.write('--Depth of receiver in grid points (depthr)\n')
        f.write(config.get('FWI','depthr')+'\n')
        f.write('--Receiver interval in grid points (dr)\n')
        f.write(config.get('FWI','dr')+'\n')
        f.write('--Time step interval of saved wavefield during forward (nt_interval)\n')
        f.write(config.get('FWI','nt_interval')+'\n')
        f.write('--Grid spacing in x direction (dx)\n')
        f.write(config.get('FWI','dx')+'\n')
        f.write('--Grid spacing in z direction (dz)\n')
        f.write(config.get('FWI','dz')+'\n')
        f.write('--Time step (dt)\n')
        f.write(config.get('FWI','dt')+'\n')
        f.write('--Donimate frequency (f0)\n')
        f.write(config.get('FWI','f0')+'\n')

    return

def run_fwi_i(theta, data, config):

    vp = theta.astype('float64')
    paramfile = config.get('FWI','configfile')
    rec, grad = fwi(vp, data, paramfile=paramfile)

    loss = np.sum((rec-data)**2)

    return loss, grad

def run_fwi(models, data, config, client=None):
    '''
    Call 2d fwi code to calculate loss and gradients
    Input
        models: velocity models, shape (n, ndim)
        data: waveform data
        config: configparser.ConfigParser()
        client: dask client to submit fwi calculation
    Return
        loss: loss value of each particle, shape (n,)
        grad: gradient of loss function w.r.t velocity, shape (n, ndim)
    '''

    # prepare configure file for 2d fwi code
    prepare_config(config)

    # submit fwi calculation to dask cluster for each model in models
    futures = []
    data_future = client.scatter(data)
    for i in range(models.shape[0]):
        futures.append( client.submit(run_fwi_i, models[i,:], data_future, config, pure=False) )

    results = client.gather(futures)

    loss = np.zeros((models.shape[0],))
    grad = np.zeros_like(models)
    for i in range(models.shape[0]):
        loss[i] = results[i][0]
        grad[i,:] = results[i][1]

    return loss, grad

