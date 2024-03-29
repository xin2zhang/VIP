import numpy as np
import time
from forward.fwi3d.run_fwi import run_fwi

class fwi3d():
    '''
    A class that implements an interface of an external 3D FWI code
    '''
    def __init__(self, config, prior, mask=None, client=None):
        '''
        config: a python configparser.ConfigParser()
        prior: a prior class, see prior/prior.py
        mask: a mask array where the parameters will be fixed, default no mask
        client: a dask client to submit fwi running, must be specified
        '''

        self.config = config
        self.sigma = config.getfloat('svgd','sigma')
        self.client = client
        self.prior = prior

        # create mask matrix for model parameters that are fixed
        nx = config.getint('svgd','nx')
        ny = config.getint('svgd','ny')
        nz = config.getint('svgd','nz')
        if(mask is None):
            mask = np.full((ny*nx*nz),False)
        self.mask = mask

    def fwi_gradient(self, theta):
        '''
        Call external FWI code to get misfit value and gradient
        Note that run_fwi needs to be implemented for specific FWI code
        '''

        # call fwi function, get loss and grad
        loss, grad = run_fwi(theta, self.config, client=self.client)
        # update grad
        grad[:,self.mask] = 0
        g = -1./(theta**3*self.sigma**2)
        grad *= g
        # clip the grad to avoid numerical instability
        clip = self.config.getfloat('FWI','gclipmax')
        #clip = clip * np.quantile(np.abs(grad),0.999)
        grad[grad>=clip] = clip
        grad[grad<=-clip] = -clip

        # log likelihood
        return 0.5*loss/self.sigma**2, grad

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

        t = time.time()
        loss, grad = self.fwi_gradient(theta)
        lglike = -loss + self.prior.lnprob(theta)
        print('Simulation takes '+str(time.time()-t))

        # compute gradient including the prior
        grad, mask = self.prior.grad(theta, grad=grad)
        grad[:,self.mask] = 0
        print(f'Average loss and negative log posterior: {np.mean(loss)} {np.mean(-lglike)}')
        print(f'Max. Mean and Median grad: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')
        #print(f'max, mean and median grads after transform: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))}')

        return lglike, grad, mask
