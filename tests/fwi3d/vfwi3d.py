#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import scipy.sparse as sparse

from vip.prior.transform import trans
from vip.fwi.fwi3d import *
from vip.pyvi.svgd import SVGD, sSVGD

from datetime import datetime
import time
from scipy.ndimage import gaussian_filter
import configparser
import vip.fwi.dask_utils as du
from vip.prior.prior import prior
from vip.prior.pdf import Uniform, Gaussian

import argparse
import sys
from pathlib import Path

def init_vfwi(config):
    Path(config.get('FWI','inpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('FWI','outpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('svgd','outpath')).mkdir(parents=True, exist_ok=True)
    Path(config.get('dask','daskpath')).mkdir(parents=True, exist_ok=True)

def generate_init(n=1, lb=0, ub=1e8, transform=True):
    eps = np.finfo(np.float).eps
    ny = lb.shape[0]; nx = lb.shape[1]; nz = lb.shape[2]
    x = lb + (ub-lb)*np.random.uniform(low=eps,high=1-eps,size=(n,ny,nx,nz))
    x = x.reshape((-1,ny*nx*nz))
    if(transform):
        x = trans(x,lb=lb.flatten(),ub=ub.flatten(),trans=1)
    return x

def get_init(config, resume=0):
    nx = config.getint('svgd','nx')
    ny = config.getint('svgd','ny')
    nz = config.getint('svgd','nz')
    priorfile = config.get('svgd','prior')
    prior = np.loadtxt(priorfile)
    lower_bnd = prior[:,0].astype(np.float64); upper_bnd = prior[:,1].astype(np.float64)
    lower_bnd = np.broadcast_to(lower_bnd[None,None,:],(ny,nx,nz))
    upper_bnd = np.broadcast_to(upper_bnd[None,None,:],(ny,nx,nz))
    if(resume==0):
        x0 = generate_init(n=config.getint('svgd','nparticles'), lb=lower_bnd, ub=upper_bnd,
                           transform=config.getboolean('svgd','transform'))
    else:
        print("Resume from previous running..")
        f = h5py.File(os.path.join(config['svgd']['outpath'],'samples.hdf5'),'r')
        x0 = f['samples'][-1,:,:]
        x0 = x0.astype(np.float64)
        f.close()
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
    nx = config.getint('svgd','nx')
    ny = config.getint('svgd','ny')
    nz = config.getint('svgd','nz')
    priorfile = config.get('svgd','prior')
    p = np.loadtxt(priorfile)
    p1 = p[:,0].astype(np.float64); p2 = p[:,1].astype(np.float64)
    p1 = np.broadcast_to(p1[None,None,:],(ny,nx,nz))
    p2 = np.broadcast_to(p2[None,None,:],(ny,nx,nz))
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
    smoothy = config.getfloat('svgd','smoothy')
    smoothz = config.getfloat('svgd','smoothz')
    if(smoothx>0 and smoothy>0 and smoothz>0):
        smoothx = np.full((nz,),fill_value=smoothx)
        smoothy = np.full((nz,),fill_value=smoothy)
        smoothz = np.full((nz,),fill_value=smoothz)
        deltax = delta(nx)
        deltay = delta(ny)
        deltaz = delta(nz)/smoothz[:,None]
        Iy = sparse.eye(ny); Ix = sparse.eye(nx); Iz = sparse.eye(nz)
        Sz = sparse.kron(Iy,sparse.kron(Ix,deltaz))
        Sx = sparse.kron(Iy,sparse.kron(deltax,Iz/smoothx))
        Sy = sparse.kron(deltay,sparse.kron(Ix,Iz/smoothy))
        L = sparse.vstack([Sx,Sy,Sz])
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
    parser.add_argument("-r", "--resume", metavar='resume', default=0, type=float, help="Resume mode (1) or start a new run(0)")

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
    cluster, client = du.dask_init(config.get('dask','pe'), config.getint('dask','nnodes'),
                                   nworkers=config.getint('dask','nworkers'),
                                   ph=config.getint('dask','ph'), odask=daskpath)
    ppdf = create_prior(config)

    # fwi simulator
    nx = config.getint('svgd','nx')
    ny = config.getint('svgd','ny')
    nz = config.getint('svgd','nz')
    mask = np.full((ny,nx,nz),False)
    mask[:,:,0] = True
    simulator = fwi3d(config, ppdf, mask=mask.flatten(), client=client)

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

    du.dask_del(cluster, client, odask=daskpath)
