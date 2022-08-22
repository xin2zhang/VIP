import numpy as np
from vfwi.pyvi.optimizer import optimizer as optm

class ADVI():
    '''
    Class ADVI()
        A class that implements ADVI (mean-field) algorithm
    '''

    def __init__(self, lnprob, kernel='meanfield', nmc=4, mu0=None, omega0=None):
        '''
        lnprob: log of the probability density function, usually negtive misfit function
        kernel: kernel function, currently only 'meanfield' supported
        optimizer: the optimizing method, including 'sgd','adagrad', 'adadelta' and 'adam'
        nmc: number of monte carlo samples used to calculate gradients
        mu0, omega0: initial mean and variance value
        '''

        self.lnprob = lnprob
        self.kernel = kernel
        self.nmc = nmc
        self.mu = mu0
        self.omega = omega0

    def grad(self, x):
        '''
        Calculate gradient for advi
        Input
            x: a vector of [mu,omega]
        Return
            loss: the loss value for all samples
            grad: gradient of x
        '''

        n = int(x.shape[0]/2)
        mu = x[0:n]
        omega = x[n:]
        eta = np.random.normal(size=(self.nmc,len(self.mu)))
        theta = mu + np.exp(omega)*eta
        loss, grad, _ = self.lnprob(theta)
        gradu = np.mean(grad,axis=0)
        gradw = grad * eta * np.exp(omega)[None,:]
        gradw = np.mean(gradw,axis=0) + 1
        grad = np.concatenate((gradu,gradw),axis=0)

        return loss, grad, None

    def sample(self, optimizer='sgd', n_iter = 1000, stepsize = 1e-3, gamma=1.0, decay_step=1, alpha=0.9,
            beta=0.99):
        '''
        Update mu and omega
        Input
            n_iter: the number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
        Return
            x: updated mu and omega
            losses: loss values across iterations
        '''

        x = np.concatenate((self.mu,self.omega),axis=0)

        op = optm(x.shape, self.grad, method=optimizer, alpha=alpha, beta=beta)
        x, losses = op.optim(x, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step)


        return x, losses
