import numpy as np
import subprocess
import os.path
import re
from pathlib import Path

def argmap(key, settings):
    return key+"="+settings[key]

def prepare_models(theta, nx, ny, nz, datapath='./', fname='particles', sep='_'):

    nparticles = theta.shape[0]
    for i in range(nparticles):
        filename = os.path.join(datapath,fname+sep+str(i)+'.npy')
        model = theta[i,:].reshape((ny,nx,nz))
        np.save(filename,model)

    return 0

def prepare_batch(batch, datapath='input', srcfile='input/source.npy', datafile='input/data.npy'):

    shots = np.load(srcfile)
    shotsize = shots.shape[0]
    slices = np.sort(np.random.choice(shotsize,size=batch,replace=False))
    batch_shots = shots[slices,:,]
    batch_src = os.path.join(datapath,'batch_src.npy')
    np.save(batch_src,batch_shots)

    data = np.load(datafile)
    batch_data = data[slices,:,:]
    batch_file = os.path.join(datapath,'batch_data.npy')
    np.save(batch_file, batch_data)

    return shotsize*1./batch

def cal_loss(pred_file, data_file='input/batch_data.npy'):
    pred_data = np.load(pred_file)
    data = np.load(data_file)
    res = pred_data - data
    loss = np.sum(res**2)

    return loss

def run_fwi_i(i, config):
    options =['waveletfile', 'recfile']
    args = []
    settings = config['FWI']
    args.append("/home/xzhang/fwi/bin/fwi")
    for op in options:
        args.append(argmap(op,settings))
    args.append("data="+os.path.join(settings['inpath'],'batch_data.npy'))
    args.append("src="+os.path.join(settings['inpath'],'batch_src.npy'))

    #print(*args, sep=" ")
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)
    except subprocess.CalledProcessError as e:
        print(e.output)
    except subprocess.TimeoutExpired:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)

    grad = np.load(os.path.join(settings['outpath'],'gradout_'+str(i)+'.npy'))
    loss = cal_loss(os.path.join(settings['outpath'],'pred_data_'+str(i)+'.npy'),
            os.path.join(settings['inpath'],'batch_data.npy'))

    return loss, grad.flatten()

def run_fwi(models, config, client=None):
    # get model info from config
    nx = config.getint('svgd','nx')
    ny = config.getint('svgd','ny')
    nz = config.getint('svgd','nz')
    dx = config.getfloat('svgd','dx')
    dy = config.getfloat('svgd','dy')
    dz = config.getfloat('svgd','dz')
    batch = config.getint('svgd','shot_batch')
    datapath=config.get('FWI','inpath')

    # prepare velocity models for FWI code (revise for sepcific code)
    prepare_models(models,ny,nx,nz,datapath=datapath)
    scale = prepare_batch(batch, datapath=datapath, srcfile=config.get('FWI','srcfile'), datafile=config.get('FWI','datafile'))

    # submit external FWI code to dask cluster for each model in models
    futures = []
    for i in range(models.shape[0]):
        futures.append( client.submit(run_fwi_i, i, config, pure=False) )

    results = client.gather(futures)

    loss = np.zeros((models.shape[0],))
    grad = np.zeros_like(models)
    for i in range(models.shape[0]):
        loss[i] = results[i][0]
        grad[i,:] = results[i][1]

    return loss, grad*scale
