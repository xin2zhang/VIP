import numpy as np
import time
from forward.tomo2d.run_tomo import run_tomo

class tomo2d():
    '''
    A class that implements an interface of an external 2d travel time tomography code
    '''
    def __init__(self, config, prior, mask=None, client=None):
        '''
        config: a python configparser.ConfigParser()
        prior: a prior class, see prior/prior.py
        mask: a mask array where the parameters will be fixed, default no mask
        client: a dask client to submit tomography running, not used here
        '''

        self.config = config
        self.sigma = config.getfloat('svgd','sigma')
        self.client = client
        self.prior = prior
        self.data = np.loadtxt(config.get('tomo','datafile'), dtype=np.float64)
        self.src = np.loadtxt(config.get('tomo','srcfile'), dtype=np.float64)
        self.rec = np.loadtxt(config.get('tomo','recfile'), dtype=np.float64)

        # create mask matrix for model parameters that are fixed
        ny = config.getint('tomo','ny')
        nx = config.getint('tomo','nx')
        if(mask is None):
            mask = np.full((nx*ny),False)
        self.mask = mask

    def gradient(self, theta):
        '''
        Call external tomography code to get misfit value and gradient
        Note that run_tomo needs to be implemented for specific tomography code
        '''

        # call fwi function, get loss and grad
        loss, grad = run_tomo(theta, self.data, self.src, self.rec, self.config, client=self.client)
        # update grad
        grad[:,self.mask] = 0
        #g = 1./self.sigma**2
        #grad *= g
        # clip the grad to avoid numerical instability
        #clip = self.config.getfloat('FWI','gclipmax')
        #grad[grad>=clip] = clip
        #grad[grad<=-clip] = -clip

        # log likelihood
        return loss, grad

    def dlnprob(self, theta):
        '''
        Compute gradient of log posterior pdf
        Input
            theta: 2D array with dimension of nparticles*nparameters
        Return
            lglike: a vector of log likelihood for each particle
            grad: each row contains the gradient for each particle
            mask: an auxilary mask array for SVGD optimization, can be safely ignored
        '''

        # adjust theta such that it is within prior or transformed back to original space
        theta = self.prior.adjust(theta)

        #t = time.time()
        lglike, grad = self.gradient(theta)
        #print('Simulation takes '+str(time.time()-t))

        # compute gradient including the prior
        grad, mask = self.prior.grad(theta, grad=grad)
        grad[:,self.mask] = 0
        print(f'Max. Mean and Median grad: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')
        #print(f'max, mean and median grads after transform: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')

        return lglike, grad, mask
