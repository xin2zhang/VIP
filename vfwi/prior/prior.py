import numpy as np
from vfwi.prior.transform import trans, trans_grad

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
        L: the smooth matrix
        '''

        self.pdf = pdf
        self.transform = transform
        self.lb = lb
        self.ub = ub
        self.smooth = smooth
        self.L = L

    def trans():
        return self.transform

    def pdf():
        return self.pdf

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
            x: value of the random variable
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
