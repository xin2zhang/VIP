#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from vip.prior.transform import trans
from forward.fwi2d.fwi2d import fwi2d
from vip.pyvi.svgd import SVGD, sSVGD

from datetime import datetime
import time
import configparser
import forward.fwi.dask_utils as du
from vip.prior.prior import prior
from vip.prior.pdf import Uniform, Gaussian

import argparse
import sys
from pathlib import Path
import dask

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['OMP_NUM_THREADS']='4'
dask.config.set({'distributed.comm.timeouts.connect': '50s'})

def init_vfwi(config):
    Path(config.get('svgd','outpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('dask','daskpath')).mkdir(parents=True, exist_ok=True)


def generate_init(n=1,lb=0,ub=1e8, transform=True):
    eps = np.finfo(np.float64).eps
    nz = lb.shape[0]; nx = lb.shape[1]
    x = lb + (ub-lb)*np.random.uniform(low=eps,high=1-eps,size=(n,nz,nx))
    x[:,0:18,:] = 1500
    if(transform):
        x = trans(x.reshape((-1,nx*nz)),lb=lb.flatten(),ub=ub.flatten(),trans=1)

    return x

def get_init(config, resume=0):
    nx = config.getint('FWI','nx')
    nz = config.getint('FWI','nz')
    priorfile = config.get('svgd','prior')
    prior = np.loadtxt(priorfile)
    lower_bnd = prior[:,0].astype(np.float64); upper_bnd = prior[:,1].astype(np.float64)
    lower_bnd = np.broadcast_to(lower_bnd[:,None],(nz,nx))
    upper_bnd = np.broadcast_to(upper_bnd[:,None],(nz,nx))
    if(resume==0):
        x0 = generate_init(n=config.getint('svgd','nparticles'), lb=lower_bnd, ub=upper_bnd,
                           transform=config.getboolean('svgd','transform'))

    else:
        print("Resume from previous running..")
        f = h5py.File(os.path.join(config['svgd']['outpath'],'samples.hdf5'),'r')
        x0 = f['samples'][-1,:,:]
        x0 = x0.astype(np.float64)
        f.close()
        if( config.getboolean('svgd','transform') ):
            x0 = trans(x0,lb=lower_bnd.flatten(),ub=upper_bnd.flatten(),trans=1)

    if(np.isinf(x0).any()): print("Error: inf occured")
    if(np.isnan(x0).any()): print("Error: nan occured")

    return x0

def delta(n):
    diag0 = np.full((n,),fill_value=-2); diag0[0]=-1; diag0[-1]=-1
    diag1 = np.full((n-1,),fill_value=1)
    diagonals = [diag0,diag1,diag1]
    D = sparse.diags(diagonals,[0,-1,1]).tocsc()
    return D

def create_prior(config):
    # prior info
    nx = config.getint('FWI','nx')
    nz = config.getint('FWI','nz')
    priorfile = config.get('svgd','prior')
    p = np.loadtxt(priorfile)
    p1 = p[:,0].astype(np.float64); p2 = p[:,1].astype(np.float64)
    p1 = np.broadcast_to(p1[:,None],(nz,nx))
    p2 = np.broadcast_to(p2[:,None],(nz,nx))
    p1 = p1.flatten()
    p2 = p2.flatten()
    ptype = config.get('svgd','priortype')
    if(ptype == 'Uniform'):
        pdf = Uniform(lb=p1, ub=p2)
    if(ptype == 'Gaussian'):
        pdf = Gaussian(mu=p1, sigma=p2)

    # create smooth matrix
    smoothness = False
    L = None
    smoothx = config.getfloat('svgd','smoothx')
    smoothz = config.getfloat('svgd','smoothz')
    if(smoothx>0 and smoothz>0):
        smoothx = np.full((nz,),fill_value=smoothx)
        smoothz = np.full((nz,),fill_value=smoothz)
        deltax = delta(nx)
        deltaz = delta(nz)/smoothz[:,None]
        Ix = sparse.eye(nx); Iz = sparse.eye(nz)
        Sx = sparse.kron(Iz/smoothx,deltax)
        Sy = sparse.kron(deltaz,Ix)
        L = sparse.vstack([Sx,Sz])
        smoothness = True

    if(ptype=='Uniform'):
        ppdf = prior(pdf=pdf, transform=config.getboolean('svgd','transform'), lb=p1, ub=p2, smooth=smoothness, L=L)
    else:
        ppdf = prior(pdf=pdf, smooth=smoothness, L=L)

    return ppdf

def write_samples(filename, pprior, start=0, chunk=10):

    if(pprior.trans):
        f = h5py.File(filename,'r+')
        samples = f['samples']
        for i in range(start,samples.shape[0]):
            samples[i,:,:] = pprior.adjust(samples[i,:,:])
        f.close()

    return 0


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Variational Full-waveform Inversion')
    parser.add_argument("-c", "--config", metavar='config', default='config.ini', help="Configuration file")
    parser.add_argument("-r", "--resume", metavar='resume', default=0, type=int, help="Resume mode (1) or start a new run(0)")

    args = parser.parse_args()
    configfile = args.config
    resume = args.resume

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Start VFWI at {current_time}...')
    print(f'Config file for VFWI is: {configfile}')

    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(configfile)

    print('Method for VFWI is: '+config['svgd']['method'])
    init_vfwi(config)
    x0 = get_init(config, resume=resume)
    print("Particles size: "+str(x0.shape))

    daskpath = config.get('dask','daskpath')
    print(f'Create dask cluster at: {daskpath}')
    cluster, client = du.dask_local(config.getint('dask','nworkers'),
                                   ph=config.getint('dask','ph'),
                                   odask=daskpath)
    ppdf = create_prior(config)

    # fwi simulator
    nx = config.getint('FWI','nx')
    nz = config.getint('FWI','nz')
    mask = np.full((nz,nx),False)
    mask[0:18,:] = True
    data = np.load(config.get('FWI','datafile'))
    #data = data.transpose(0,2,1).flatten()
    simulator = fwi2d(config, ppdf, data, mask=mask.flatten(), client=client)

    # svgd sampler
    stepsize = config.getfloat('svgd','stepsize')
    iteration = config.getint('svgd','iter')
    final_decay = config.getfloat('svgd','final_decay')
    gamma = final_decay**(1./iteration)
    if(config['svgd']['method']=='ssvgd'):
        svgd = sSVGD(simulator.dlnprob,
                     kernel=config['svgd']['kernel'],
                     mask=mask.flatten(),
                     out=os.path.join(config.get('svgd','outpath'),'samples.hdf5'))
    elif(config['svgd']['method']=='svgd'):
        svgd = SVGD(simulator.dlnprob,
                    kernel=config['svgd']['kernel'],
                    mask=mask.flatten(),
                    out=os.path.join(config.get('svgd','outpath'),'samples.hdf5'))
    else:
        print('Not supported method')

    # sampling
    print('Start sampling ...')
    print(f'Iteration: {iteration}')
    print(f'Stepsize, decay rate and final decay: {stepsize} {gamma} {final_decay}')
    start = time.time()
    losses = svgd.sample(x0,
                    n_iter=config.getint('svgd','iter'),
                    stepsize=stepsize, gamma=gamma,
                    optimizer=config['svgd']['optimizer'],
                    burn_in=config.getint('svgd','burn_in'),
                    thin=config.getint('svgd','thin')
                    )
    end=time.time()
    print('Time taken: '+str(end-start)+' s')

    # write out results
    write_samples(os.path.join(config.get('svgd','outpath'),'samples.hdf5'), ppdf)
    with open(os.path.join(config.get('svgd','outpath'),'misfits.txt'),"ab") as f:
        np.savetxt(f,losses)

    du.dask_del(cluster, client)
