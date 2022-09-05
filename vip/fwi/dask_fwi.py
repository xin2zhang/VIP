import numpy as np
import subprocess
import PyDDS.dds_io as io
import os.path
import re
from pathlib import Path

def argmap(key, settings):
    return key+"="+settings[key]

def prepare_models(theta, ny, nx, nz, dx, dy, dz, datapath='./', fname='particles', sep='.'):

    nparticles = theta.shape[0]
    for i in range(nparticles):
        filename = os.path.join(datapath,fname+sep+str(i))
        model = theta[i,:].reshape((ny,nx,nz))
        io.tofile(filename,model,axes=['z','x','y'],units=['m','m','m'],origins=[0,0,0],deltas=[dz,dx,dy],bases=[1,1,1],steps=[1,1,1])

    return 0

def prepare_batch(batch, datapath='input', geomfile='input/Hgeom',datafile='input/data'):

    geom = io.fromfile(geomfile)['Samples']
    geom_dict = io.DDSInputDict('',geomfile)
    shotsize = geom_dict.axis3.size
    slices = np.sort(np.random.choice(shotsize,size=batch,replace=False))
    geom = geom[slices,:,:]
    batch_geom = os.path.join(datapath,'Hgeom.batch')
    io.tofile(batch_geom,geom)

    data = io.fromfile(datafile)['Samples']
    data = data[slices,:,:]
    data_dict = io.DDSInputDict('',datafile)
    batch_data = os.path.join(datapath,'data.batch')
    io.tofile(batch_data, data,origins=[data_dict.axis1.origin,0,0],deltas=[data_dict.axis1.delta,1,1])
    geom_dict.fclose()
    data_dict.fclose()

    return shotsize*1./batch

def read_loss(filename):
    # read loss from output of external FWI code (revise as needed)
    with open(filename) as f:
        text = f.read()
    numeric_const_pattern = '( [-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ? )'
    pattern = 'iter,f,gnorm,tevalFG:\s*\d+\s*'+numeric_const_pattern
    rx = re.compile(pattern,re.VERBOSE)
    results = rx.findall(text)

    loss = -1.
    if(results):
        loss = float(results[-1])

    return loss

def cal_loss(filename):
    data_dict = io.DDSInputDict('',filename)
    rd = io.DDSFileReader(data_dict)
    res = rd[0,:]['Samples']
    loss = np.sum(res**2)

    return loss

def run_fwi_i(i, config):
    options =['np','nproc','ph','imag','usedatmsk','tmp_data_path','bindata','gather_resamp',
              'nxaper','nyaper','Tmax_source','gathertype','operator','isogradient',
              'invtype','bctaper','Tmin_rcvr','Tmax_rcvr','nxtaper','nztaper',
              'nytaper','filterUpCrap','optype','illumOption','tdfwimod',
              'niter','kout','betascale','perturb_ls','vclipminvol','vclipmaxvol',
              'abs_surf','srcwpw','lessio','offmin','offmax',
              'aper_max_x','aper_max_y']
    args = []
    settings = config['FWI']
    #args.append("/tstapps/asi/bin/tdwi")
    args.append("/lustre03/other/EIP/FWI_CODE/code_v2/tdwi")
    for op in options:
        args.append(argmap(op,settings))
    args.append("geom="+os.path.join(settings['inpath'],'Hgeom.batch'))
    args.append("in="+os.path.join(settings['inpath'],'data.batch'))
    args.append("datmsk="+settings['datmskfile'])
    args.append("source="+settings['srcfile'])
    args.append("data_path="+settings['outpath'])
    args.append("print_path="+os.path.join(settings['print_path'],'printout.'+str(i)))
    args.append("vel="+os.path.join(settings['inpath'],'particles.'+str(i)))
    args.append("velout="+os.path.join(settings['outpath'],'velout.'+str(i)))
    args.append("grad="+os.path.join(settings['outpath'],'gradout.'+str(i)))
    args.append("plotfile="+os.path.join(settings['outpath'],'plot.'+str(i)))
    args.append("ures="+os.path.join(settings['outpath'],'ures.'+str(i)))
    args.append("urestmp="+os.path.join(settings['outpath'],'urestmp.'+str(i)))
    args.append("uvsrc="+os.path.join(settings['outpath'],'uvsrc.'+str(i)))
    args.append("ucalc="+os.path.join(settings['outpath'],'ucalc.'+str(i)))

    #print(*args, sep=" ")
    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)
    except subprocess.CalledProcessError as e:
        print(e.output)
    except subprocess.TimeoutExpired:
        subprocess.check_output(args, stderr=subprocess.STDOUT, shell=False, timeout=1200)

    grad = io.fromfile(os.path.join(settings['outpath'],'gradout.'+str(i)))['Samples']
    loss = read_loss(os.path.join(settings['print_path'],'printout.'+str(i)))
    if(loss<0):
        loss = cal_loss(os.path.join(settings['outpath'],'ures.'+str(i)))

    return loss, -grad.flatten()

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
    prepare_models(models,ny,nx,nz,dx,dy,dz,datapath=datapath)
    scale = prepare_batch(batch, datapath=datapath, geomfile=config.get('FWI','geomfile'), datafile=config.get('FWI','datafile'))

    for filename in Path(config.get('FWI','outpath')).glob("*_restart.state*"):
        filename.unlink()

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

def read_many_data(x, datapath='./', fname='gradout', sep='.'):

    for i in range(x.shape[0]):
        data = io.fromfile(os.path.join(datapath,fname+sep+str(i)))['Samples']
        x[i,:] = data.flatten()

    return x

def read_many_loss(n=1, datapath='./', fname='tdfwi.small', sep='.'):
    # read loss from output of external FWI code (revise as needed)
    loss = np.zeros((n,))
    for i in range(n):
        filename = os.path.join(datapath,fname+sep+str(i))
        with open(filename) as f:
            text = f.read()
        numeric_const_pattern = '( [-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ? )'
        pattern = 'iter,f,gnorm,tevalFG:\s*\d+\s*'+numeric_const_pattern
        rx = re.compile(pattern,re.VERBOSE)
        results = rx.findall(text)
        loss[i] = float(results[-1])

    return loss
