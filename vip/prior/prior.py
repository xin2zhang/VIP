import numpy as np
from vip.prior.transform import trans, trans_grad
from vip.kernel.pytrans import pyjac

class prior():
    '''
    Class prior()
        A class that implements prior information
    '''

    def __init__(self, pdf=None, transform=False, lb=None, ub=None,
                 smooth=False, L=None):
        '''
        pdf: a pdf class defined in pdf.py
        transform: if True, transform from constrained variable to unconstrained variable
        lb, ub: if transform is True, specify the lower and upper boundary
        smooth: if True, adding smoothness constraint
        L: smooth matrix which is a scipy sparse matrix
        '''

        self.pdf = pdf
        self.transform = transform
        self.lb = lb
        self.ub = ub
        self.smooth = smooth
        self.L = L

    def trans(self):
        return self.transform

    def pdf(self):
        return self.pdf

    def lnprob(self, x):
        '''
        Calculate log probability of x
        Input
            x: value of the random variable, shape (nparticles,nparameters)
        Return
            logp: the log probability
        '''

        logp = self.pdf.lnprob(x)

        if(self.smooth):
            logsp = self.logp_smooth(x)
            logp = logp + logsp

        if(self.transform):
            logp = logp + pyjac(x,self.lb,self.ub)

        return logp

    def logp_smooth(self,x):
        ''' return smooth log probability
        Input
            x: 2D array with dimension nparticles*nparameters
        Return: log probability, shape (nparticles,)
        '''
        dx = self.L*x.T
        self.L = self.L.eliminate_zeros()
        num_elements = self.L.indptr[1:] - self.L.indptr[:-1]
        diag = self.L.diagonal()/(num_elements-1)
        logsp = -0.5*np.sum(dx**2,axis=0) + np.sum(np.log(np.abs(diag))) - 0.5*self.L.shape[0]*np.log(2*np.pi)

        return logsp

    def grad_matrix(self, x, grad):
        ''' return gradient of smooth prior
        Input
            grad: grad will be updated in-place for saving memory
            x: 2D array with dimension nparticles*nparameters
        Return: updated gradient
        '''
        g = self.L*x.T
        g = self.L.transpose()*g
        grad += -g.T

        return grad

    def grad(self, x, grad=None, chunks=None):
        '''
        Calculate gradient of log probability w.r.t x
        Input
            x: value of the random variable, 2D array with dimension nparticles*nparameters
            grad: grad contains the gradient of likelihood w.r.t original variable
        Return: gradient including prior pdf
        '''

        if(grad is None):
            grad = np.zeros(x.shape)

        if(self.smooth):
            grad = self.grad_matrix(x, grad)

        mask = np.zeros(x.shape,dtype=bool)
        if(self.transform):
            x = trans(x, batchsize=chunks, trans=1, lb=self.lb, ub=self.ub)
            grad, mask = trans_grad(grad, x, mask, batchsize=chunks, lb=self.lb, ub=self.ub)
        else:
            g = self.pdf.grad(x)
            grad += g

        return grad, mask

    def clip(self, x):
        '''
        Clip the variable according to the lower and upper boundary if transform is False
        Input: x is the variable
        Return: clipped x
        '''
        w = np.where(x < self.lb)
        x[w] = self.lb[w[1]]
        w = np.where(x > self.ub)
        x[w] = self.ub[w[1]]

        return x

    def adjust(self, x, chunk=None):
        '''
        Adjust variable to be within prior if Uniform, or transform back to original variable if transform is True
        Input: x
        Return: adjusted x
        '''

        if( (not self.transform) and (self.lb is not None) ):
            x = self.clip(x)

        if( self.transform ):
            x = trans(x, batchsize=chunk, trans=0, lb=self.lb, ub=self.ub)

        return x
