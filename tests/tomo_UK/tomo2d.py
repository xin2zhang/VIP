#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from vip.prior.transform import trans
from forward.tomo.tomo2d import tomo2d
from vip.pyvi.svgd import SVGD, sSVGD

from datetime import datetime
import time
import configparser
#import vip.fwi.dask_utils as du
from vip.prior.prior import prior
from vip.prior.pdf import Uniform, Gaussian

import argparse
import sys
from pathlib import Path

os.environ['HDF5_USE_FILE_LOCKING']='FALSE'
os.environ['OMP_NUM_THREADS'] = '40'

def init_tomo(config):
    Path(config.get('svgd','outpath')).mkdir(parents=True, exist_ok=True)


def generate_init(n=1,lb=0,ub=1e8, transform=True):
    eps = np.finfo(np.float).eps
    x = lb + (ub-lb)*np.random.uniform(low=eps,high=1-eps,size=(n,lb.shape[0]))
    if(transform):
        x = trans(x.reshape((-1,lb.shape[0])),lb=lb, ub=ub, trans=1)

    return x

def get_init(config, resume=0):
    priorfile = config.get('svgd','prior')
    prior = np.loadtxt(priorfile)
    lower_bnd = prior[:,0].astype(np.float64); upper_bnd = prior[:,1].astype(np.float64)
    if(resume==0):
        x0 = generate_init(n=config.getint('svgd','nparticles'), lb=lower_bnd, ub=upper_bnd,
                           transform=config.getboolean('svgd','transform'))

    else:
        print("Resume from previous running..")
        #f = h5py.File(os.path.join(config['svgd']['outpath'],'samples.hdf5'),'r')
        x0 = np.load('last_sample.npy')
        w = np.where(x0<=lower_bnd)
        for i in range(len(w[0])):
            x0[w[0][i],w[1][i]] = np.mean(x0[:,w[1][i]])
        w = np.where(x0>=upper_bnd)
        for i in range(len(w[0])):
            x0[w[0][i],w[1][i]] = np.mean(x0[:,w[1][i]])
        x0 = x0.astype(np.float64)
        #f.close()

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
    priorfile = config.get('svgd','prior')
    p = np.loadtxt(priorfile)
    p1 = p[:,0].astype(np.float64); p2 = p[:,1].astype(np.float64)
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
    if(smoothx>0 and smoothy>0):
        smoothx = np.full((ny,),fill_value=smoothx)
        smoothz = np.full((ny,),fill_value=smoothy)
        deltax = delta(nx)
        deltay = delta(ny)/smoothy[:,None]
        Ix = sparse.eye(nx); Iy = sparse.eye(ny)
        Sx = sparse.kron(Iy/smoothx,deltax)
        Sy = sparse.kron(deltay,Ix)
        L = sparse.vstack([Sx,Sy])
        smoothness = True

    if(ptype=='Uniform'):
        ppdf = prior(pdf=pdf, transform=config.getboolean('svgd','transform'), lb=p1, ub=p2, smooth=smoothness, L=L)
    else:
        ppdf = prior(pdf=pdf, smooth=smoothness, L=L)

    return ppdf

def write_samples(filename, pprior, n=0, chunk=10):

    f = h5py.File(filename,'r+')
    samples = f['samples']
    start = 0
    if(n>0):
        start = samples.shape[0] - n
    if(start<0):
        start = 0
    if(pprior.trans):
        for i in range(start,samples.shape[0]):
            samples[i,:,:] = pprior.adjust(samples[i,:,:])

    mean = np.mean(samples[:].reshape((-1,samples.shape[2])),axis=0)
    std = np.std(samples[:].reshape((-1,samples.shape[2])),axis=0)
    last = samples[-1,:,:]
    f.close()

    np.save('mean.npy',mean)
    np.save('std.npy',std)
    np.save('last_sample.npy',last)
    return 0


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Variational Tomography')
    parser.add_argument("-c", "--config", metavar='config', default='config.ini', help="Configuration file")
    parser.add_argument("-r", "--resume", metavar='resume', default=0, type=float, help="Resume mode (1) or start a new run(0)")

    args = parser.parse_args()
    configfile = args.config
    resume = args.resume

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f'Start VTomo at {current_time}...')
    print(f'Config file for VFWI is: {configfile}')

    config = configparser.ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(configfile)

    print('Method for VTomo is: '+config['svgd']['method'])
    init_tomo(config)
    x0 = get_init(config, resume=resume)
    print("Particles size: "+str(x0.shape))

    ppdf = create_prior(config)

    # fwi simulator
    simulator = tomo2d(config, ppdf)

    # svgd sampler
    stepsize = config.getfloat('svgd','stepsize')
    iteration = config.getint('svgd','iter')
    final_decay = config.getfloat('svgd','final_decay')
    gamma = final_decay**(1./iteration)
    if(config['svgd']['method']=='ssvgd'):
        svgd = sSVGD(simulator.dlnprob,
                     kernel=config['svgd']['kernel'],
                     out=os.path.join(config.get('svgd','outpath'),'samples.hdf5'))
    elif(config['svgd']['method']=='svgd'):
        svgd = SVGD(simulator.dlnprob,
                    kernel=config['svgd']['kernel'],
                    h = 1.0,
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
                    pre_update=0, pre_step=1e-3,
                    optimizer=config['svgd']['optimizer'],
                    burn_in=config.getint('svgd','burn_in'),
                    thin=config.getint('svgd','thin'),
                    )
    end=time.time()
    print('Time taken: '+str(end-start)+' s')

    # write out results
    nsamples = int((config.getint('svgd','iter')-config.getint('svgd','burn_in'))/config.getint('svgd','thin'))
    write_samples(os.path.join(config.get('svgd','outpath'),'samples.hdf5'), ppdf, n=nsamples)
    with open(os.path.join(config.get('svgd','outpath'),'misfits.txt'),"ab") as f:
        np.savetxt(f,losses)
