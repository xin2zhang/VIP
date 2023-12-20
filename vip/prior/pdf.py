import numpy as np

class pdf():
    '''
    Class pdf()
        A class implements probability density functions
    '''
    def __init__(self,):
        pass

    def lnprob(self, theta):
        '''
        Compute log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the log probability
        '''
        pass

    def grad(self, theta):
        '''
        Compute gradient of log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the gradient
        '''
        pass

class Uniform():
    '''
    Class Uniform()
        A class that implements Uniform distribution
    '''

    def __init__(self, lb=0, ub=1):
        '''
        lb, ub: the lower and upper bound
        '''
        self.lb = lb
        self.ub = ub

    def lb(self):
        return self.lb

    def ub(self):
        return self.ub

    def lnprob(self, theta):
        '''
        Compute log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the log probability
        '''

        logp = -np.sum(np.log(self.ub-self.lb))
        if(np.any(theta<self.lb) or np.any(theta>self.ub)):
             logp = -np.inf

        return logp

    def grad(self, theta):
        '''
        Compute gradient of log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the gradient
        '''
        g = 0

        return g

class Gaussian():
    '''
    Class Gaussian()
        A class that implements Gaussian distribution
    '''

    def __init__(self, mu=0, sigma=1, cov=None):
        '''
        mu: mean
        sigma: standard deviation
        '''
        self.mu = mu
        self.sigma = sigma

    def mean(self):
        return self.mu

    def std(self):
        return self.sigma

    def cov(self):
        return self.cov

    def lnprob(self, theta):
        '''
        Compute log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the log probability
        '''

        logp = -0.5*np.sum(((theta-self.mu)/self.sigma)**2) - np.sum(np.log(self.sigma)) - 0.5*self.mu.shape[0]*np.log(2*np.pi)

        return logp

    def grad(self, theta):
        '''
        Compute gradient of log probability
        Input
            theta: the value of random variable
        Return
            lnprob: the gradient
        '''
        g = -(theta-self.mu)/self.sigma**2

        return g
