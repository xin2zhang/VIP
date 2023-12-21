import numpy as np
import scipy.linalg as sla
from vip.kernel.pykernel import *
import vip.pyvi.optimizer as optm
import h5py
import os.path

import time

def svgd_grad(x, grad, kernel='rbf', w=None, h=1.0, chunks=None):
    ''' Calculate gradient for svgd update
        Input
        x - 2d array of particles with size (nparticles,nparameters)
        grad - 2d array of each particle gradient, same size as x
        kernel - kernel function, only "rbf", "diagonal" supported for now
        w - diagonal elements for diagonal matrix kernel, is a vector of size nparameters
        h - bandwith for rbf kernel function, is a frac of median trick bandwith.
            h is positive, actual bandwith is h*medh
        chunk - batchsize for number of particles, if None, batchsize equals nparticles
        Output
        kxy - kernel matrix, size (nparticles,nparticles)
        grad - svgd gradient, size (particles,nparameters)
    '''

    ndim = x.shape[1]
    if(w is None):
        w = np.full((ndim,),fill_value=1.0)
    if(chunks is None):
        chunks = x.shape
    dist = 0
    for i in range(0,ndim,chunks[1]):
        end = min(i+chunks[1],ndim)
        dist = dist + pdist(np.ascontiguousarray(x[:,i:end]),np.ascontiguousarray(w[i:end]))

    for i in range(0,ndim,chunks[1]):
        end = min(i+chunks[1],ndim)
        grad[:,i:end] = svgd_gradient(np.ascontiguousarray(x[:,i:end]),
                                      np.ascontiguousarray(grad[:,i:end]),
                                      dist,w[i:end],kernel=kernel,h=h)
    medh = np.median(dist)
    medh = np.sqrt(0.5*medh/np.log(x.shape[0]+1))
    h = h*medh
    kxy = np.exp(-dist/(2*h**2))

    return kxy, grad

class SVGD():
    '''
    A class that implements SVGD algorithm
    '''

    def __init__(self, lnprob, kernel='rbf', h=1.0, weight='grad', mask=None,
                 threshold=0.02, out='samples.hdf5'):
        '''
        lnprob: log of the probability density function, usually negtive misfit function
        kernel: kernel function, including rbf and diagonal matrix kernel
        h:  bandwith for rbf kernel function, is a frac of median trick bandwith.
            h is positive, actual bandwith is h*medh
        weight: method of coonstructing diagonal matrix, 'var' using variance of each parameters across particles,
            'grad' using 1/sqrt(grad**2) similar as in adagrad, 'delta' using sqrt(dg**2)/sqrt(dm**2)
        mask: mask array where the variables are fixed, i.e. the gradient is zero, default no mask
        out: hdf5 file that stores final particles

        '''

        self.h = h
        self.lnprob = lnprob
        self.kernel = kernel
        self.weight = weight
        self.mask = mask
        self.threshold = threshold
        self.out = out

    def sample(self, x0, optimizer='sgd', n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               alpha=0.9, beta=0.95, burn_in=None, thin=None, chunks=None):
        '''
        Using svgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            optimizer: optimization method, including 'sgd', 'adam', default to 'sgd'
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            burn_in, thin: not used, just for consistent arguments
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: loss value for each particle at each iteration, shape(n_iter, n)
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        if(x0 is None):
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        theta = x0

        if(self.kernel=='rbf'):
            op = optm.optimizer(x0.shape, self.grad, method=optimizer, alpha=alpha, beta=beta)
            theta, losses = op.optim(theta, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step)

        if(self.kernel=='diagonal'):
            theta, losses = self.__dsample(theta, n_iter=n_iter, stepsize=stepsize, gamma=gamma,
                                           decay_step=decay_step, alpha=alpha, chunks=chunks)

        f = h5py.File(self.out,'w')
        samples = f.create_dataset('samples',(1,x0.shape[0],x0.shape[1]),
                                   compression="gzip", chunks=True)
        samples[0,:,:] = np.copy(theta)
        f.close()

        return losses

    def grad(self, theta, mkernel=None, chunks=None):
        '''
        Compute gradients for svgd update
        Input
            theta: the current value of variable (transformed), shape (n,dim)
            mkernel: the vector of the diagonal matrix with length dim, if using a diagonal matrix kernel
            chunks: chunks of particles for calculation, default n
        Return
            loss: a vector of loss value of all particles
            grad: svgd gradients for each particles, shape (n,dim)
            mask: a mask array
        '''
        if(chunks is None):
            chunks = theta.shape

        logp, grad, mask = self.lnprob(theta)
        if(mask is None):
            mask = np.full((theta.shape),fill_value=False)
        if(mkernel is None):
            mkernel = np.full((theta.shape[1],),fill_value=1.0)

        #ksd, stepsize = pyksd(theta, grad, mkernel, kernel=self.kernel, h=self.h)
        kxy, grad = svgd_grad(theta, grad, kernel=self.kernel, w=mkernel, h=self.h, chunks=chunks)
        print(f'max, mean, median, and min grads for svgd: {np.max(abs(grad))} {np.mean(abs(grad))} {np.median(abs(grad))} {np.min(abs(grad))}')
        #print('Average loss: '+str(np.mean(-loss)))

        return -logp, grad, mask

    def __dsample(self, theta, n_iter=100, stepsize=1e-2, gamma=1.0, decay_step=1, alpha=0.9, chunks=None):

        # initialise some variables
        losses = np.zeros((n_iter,theta.shape[0]))
        prev_grad = np.zeros(theta.shape,dtype=np.float64)
        prev_theta = np.zeros(theta.shape,dtype=np.float64)
        mkernel = np.full((theta.shape[1],),fill_value=1.0, dtype=np.float64)
        w = weight(dim=theta.shape[1], approx=self.weight, threshold=self.threshold)
        moment = np.zeros(theta.shape,dtype=np.float64)

        # sampling
        for i in range(n_iter):

            # start a new iteration
            print(f'Iteration: {i}')
            #print(f'max, mean, median and min kernel: {np.max(abs(mkernel))} {np.mean(abs(mkernel))} {np.median(abs(mkernel))} {np.min(abs(mkernel))}')
            print(f'max, mean, median and min theta: {np.max(abs(theta))} {np.mean(abs(theta))} {np.median(abs(theta))} {np.min(abs(theta))}')
            loss, grad, mask = self.grad(theta, mkernel=mkernel, chunks=chunks)

            mkernel = w.diag(theta, prev_theta, grad, prev_grad)
            prev_grad = np.copy(grad)
            prev_theta = np.copy(theta)

            moment[mask] = 0
            moment = alpha*moment + stepsize*grad
            theta += moment
            losses[i,:] = loss
            #print('Average loss: '+str(np.mean(loss)))

            # decay the stepsize if required
            if((i+1)%decay_step == 0):
                stepsize = stepsize * gamma

        return theta, losses


class sSVGD():
    '''
    A class that implements stochastic SVGD algorithm.
    '''
    def __init__(self, lnprob, kernel='rbf', h=1.0, metropolis=False, mask=None, threshold=0.02,
                 weight='grad', out='samples.hdf5'):
        '''
        lnprob: log of the probability density function, usually negtive misfit function
        kernel: kernel function, including rbf and diagonal matrix kernel
        h:  bandwith for rbf kernel function, is a frac of median trick bandwith.
            h is positive, actual bandwith is h*medh
        weight: method of coonstructing diagonal matrix, 'var' using variance of each parameters across particles,
            'grad' using 1/sqrt(grad**2) similar as in adagrad, 'delta' using sqrt(dg**2)/sqrt(dm**2)
        mask: mask array where the variables are fixed, i.e. the gradient is zero, default no mask
        out: hdf5 file that stores final particles

        '''

        self.h = h
        self.lnprob = lnprob
        self.kernel = kernel
        self.metropolis = metropolis
        self.mask = mask
        self.out = out
        self.threshold = threshold
        self.weight = weight
        if(kernel=='rbf'):
            self.weight = 'constant'

    def sample(self, x0, n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            burn_in: burn_in period
            thin: thining of the chain
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: loss value for each iterations, shape (n_iter,n)
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        # create a hdf5 file to store samples on disk
        nsamples = int((n_iter-burn_in)/thin)
        if(not os.path.isfile(self.out)):
            f = h5py.File(self.out,'a')
            samples = f.create_dataset('samples',(nsamples,x0.shape[0],x0.shape[1]),
                                       maxshape=(None,x0.shape[0],x0.shape[1]),
                                       compression="gzip", chunks=True)
        else:
            f = h5py.File(self.out,'a')
            f['samples'].resize((f['samples'].shape[0]+nsamples),axis=0)
            samples = f['samples']

        # sampling
        if(self.metropolis):
            losses = self.ma_sample(x0, samples, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step,
               burn_in=burn_in, thin=thin, alpha=alpha, beta=beta, chunks=chunks, optimizer=optimizer)
        else:
            losses = self.pl_sample(x0, samples, n_iter=n_iter, stepsize=stepsize, gamma=gamma, decay_step=decay_step,
               burn_in=burn_in, thin=thin, alpha=alpha, beta=beta, chunks=chunks, optimizer=optimizer)

        # close hdf5 file
        f.close()

        return losses

    def grad(self, theta, mkernel=None, chunks=None):
        '''
        Compute gradients for ssvgd update
        Input
            theta: the current value of variable (transformed), shape (n,dim)
            mkernel: the vector of the diagonal matrix with length dim, if using a diagonal matrix kernel
            chunks: chunks of theta for calculation, default theta.shape
        Return
            logp: log posterior pdf value across particles
            sgrad: svgd gradients for each particles, shape (n,dim)
            kxy: kernel matrix, shape (n,n)
        '''

        if(mkernel is None):
            mkernel = np.full((theta.shape[1],),fill_value=1.0)

        logp, grad, _ = self.lnprob(theta)
        kxy, sgrad = svgd_grad(theta, grad, kernel=self.kernel, w=mkernel, h=self.h, chunks=chunks)
        print(f'max, mean, median, and min grads for svgd: {np.max(abs(sgrad))} {np.mean(abs(sgrad))} {np.median(abs(sgrad))} {np.min(abs(sgrad))}')

        return logp, sgrad, kxy

    def update(self, theta, step=1e-3, mkernel=None, chunks=None):
        '''
        Compute gradients for ssvgd update
        Input
            theta: the current value of variable (transformed), shape (n,dim)
            mkernel: the vector of the diagonal matrix with length dim, if using a diagonal matrix kernel
            chunks: chunks of theta for calculation, default theta.shape
        Return
            update_step: update at each iteration
            loss: mean loss value across particles
            grad: svgd gradients for each particles, shape (n,dim)
        '''

        if(mkernel is None):
            mkernel = np.full((theta.shape[1],),fill_value=1.0)

        # get svgd gradient and kernel matrix K
        logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)

        # calculate cholesky decomposition of kernel matrix K and generate random variable
        cholK = np.linalg.cholesky(2*kxy/theta.shape[0])
        random_update = np.sqrt(1./mkernel)*np.matmul(cholK,np.random.normal(size=theta.shape))

        update_step = step*sgrad + np.sqrt(step)*random_update
        if(self.mask is not None):
            update_step[:,self.mask] = 0

        return update_step, logp, sgrad

    def pl_sample(self, x0, samples, n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            burn_in: burn_in period
            thin: thining of the chain
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: loss value for each iterations, shape (n_iter,n)
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        theta = np.copy(x0).astype(np.float64)
        losses = np.zeros((n_iter,x0.shape[0]))

        # initialise some variables
        nsamples = int((n_iter-burn_in)/thin)
        sample_count = 0
        prev_grad = np.zeros(x0.shape,dtype=np.float64)
        prev_theta = np.zeros(x0.shape,dtype=np.float64)
        mkernel = np.full((theta.shape[1],),fill_value=1.0, dtype=np.float64)
        w = weight(dim=theta.shape[1], approx=self.weight, threshold=self.threshold)

        # sampling
        for i in range(n_iter):

            # start a new iteration
            print(f'Iteration: {i}')
            #print(f'max, mean, median and min kernel: {np.max(abs(mkernel))} {np.mean(abs(mkernel))} {np.median(abs(mkernel))} {np.min(abs(mkernel))}')
            print(f'max, mean, median and min theta: {np.max(abs(theta))} {np.mean(abs(theta))} {np.median(abs(theta))} {np.min(abs(theta))}')
            update_step, logp, pgrad = self.update(theta, step=stepsize, mkernel=mkernel, chunks=chunks)

            mkernel = w.diag(theta, prev_theta, pgrad, prev_grad)
            prev_grad = np.copy(pgrad)
            prev_theta = np.copy(theta)

            theta = theta + update_step
            losses[i,:] = -logp
            #print('Average loss: '+str(np.mean(-logp)))

            # decay the stepsize if required
            if((i+1)%decay_step == 0):
                stepsize = stepsize * gamma

            # after burn_in then collect samples
            if(i>=burn_in and (i-burn_in)%thin==0):
                samples[-nsamples+sample_count,:,:] = np.copy(theta)
                sample_count += 1

        return losses

    def ma_sample(self, x0, samples, n_iter=1000, stepsize=1e-2, gamma=1.0, decay_step=1,
               burn_in=100, thin=2, alpha=0.9, beta=0.95, chunks=None, optimizer=None):
        '''
        Using ssvgd to sample a probability density function
        Input
            x0: initial value, shape (n,dim)
            n_iter: number of iterations
            stepsize: stepsize for each iteration
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the stepsize
            burn_in: burn_in period
            thin: thining of the chain
            alpha, beta: hyperparameter for sgd and adam, for sgd only alpha is ued
            chunks: chunks of theta for calculation, default theta.shape
        Return
            losses: mean loss value for each iterations, vector of length n
            The final particles are stored at the hdf5 file specified by self.out, so no return samples
        '''

        # Check input
        if x0 is None :
            raise ValueError('x0 cannot be None!')

        if(chunks is None):
            chunks = x0.shape

        theta = np.copy(x0).astype(np.float64)
        losses = np.zeros((n_iter,x0.shape[0]))

        # initialise some variables
        nsamples = int((n_iter-burn_in)/thin)
        sample_count = 0; accepted_count = 0
        mkernel = np.full((theta.shape[1],),fill_value=1.0, dtype=np.float64)
        logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)
        cholK = np.linalg.cholesky(2*kxy/theta.shape[0])

        # save computed info for the current model
        prev_logp = logp
        prev_grad = sgrad
        prev_kxy = kxy
        prev_cholK = cholK
        prev_theta = theta

        # sampling
        for i in range(n_iter):

            # start a new iteration
            print(f'Iteration: {i}')
            print(f'max, mean, median and min theta: {np.max(abs(theta))} {np.mean(abs(theta))} {np.median(abs(theta))} {np.min(abs(theta))}')

            # update on the current model
            random_update = np.sqrt(1./mkernel)*np.matmul(cholK,np.random.normal(size=theta.shape))
            update_step = stepsize*sgrad + np.sqrt(stepsize)*random_update
            if(self.mask is not None):
                update_step[:,self.mask] = 0

            # update theta
            theta = theta + update_step

            # compute forward proposal pdf q(theta_k+1|theta_k)
            update = theta - prev_theta - stepsize*sgrad # masked variables have no effect on proposal pdf
            pvar = sla.solve_triangular(cholK,update)
            forward_plogp = -0.25/stepsize*np.sum(pvar.flatten()**2) - 0.5*sla.det(2*kxy/theta.shape[0]) - 0.5*pvar.flatten().size*np.log(2*np.pi)

            # compute info for updated theta
            logp, sgrad, kxy = self.grad(theta, mkernel=mkernel, chunks=chunks)

            # compute reverse proposal pdf q(theta_k|theta_k+1)
            cholK = np.linalg.cholesky(2*kxy/theta.shape[0])
            reverse_update = prev_theta - theta - stepsize*sgrad
            pvar = sla.solve_triangular(cholK,reverse_update)
            reverse_plogp = -0.25/stepsize*np.sum(pvar.flatten()**2) - 0.5*sla.det(2*kxy/theta.shape[0]) - 0.5*pvar.flatten().size*np.log(2*np.pi)

            # compute acceptance ratio
            acceptance_ratio = np.sum(logp) + reverse_plogp - ( np.sum(prev_logp) + forward_plogp )
            print(f'log pdf change: {np.sum(logp-prev_logp)} {reverse_plogp-forward_plogp}')
            acceptance_ratio = min(0,acceptance_ratio)

            if( np.log(np.random.uniform()) < acceptance_ratio ):
                # update previously save info if accepted
                prev_theta = theta
                prev_logp = logp
                prev_grad = sgrad
                prev_kxy = kxy
                prev_cholK = cholK
                accepted_count = accepted_count + 1
            else:
                # recover info if rejected
                theta = prev_theta
                kxy = prev_kxy
                cholK = prev_cholK
                logp = prev_logp
                sgrad = prev_grad


            losses[i,:] = -logp
            print(f'Real time and total acceptance rate: {np.exp(acceptance_ratio)} {accepted_count*1.0/(i+1)}')

            # decay the stepsize if required
            if((i+1)%decay_step == 0):
                stepsize = stepsize * gamma

            # after burn_in then collect samples
            if(i>=burn_in and (i-burn_in)%thin==0):
                samples[-nsamples+sample_count,:,:] = np.copy(theta)
                sample_count += 1

        return losses

class weight():
    '''
    A class that generates a weigting vector for kernel functions
    '''

    def __init__(self, dim=100, approx='grad', alpha=0.95, beta=0.9,
                 quantile=0.8, threshold=0.02):

        self.dm = np.zeros((dim,),dtype=np.float64)
        self.dg = np.zeros((dim,),dtype=np.float64)
        self.approx = approx
        self.kernel = np.full((dim,),fill_value=1.0,dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.quantile = quantile
        self.threshold = threshold

    def diag(self, cm, pm, cg, pg, eps=1E-5):

        if(self.approx=='constant'):
            pass

        elif(self.approx=='var'):
            kernel = 1/(np.var(cm,axis=0)+eps)
            kernel[kernel<self.threshold] = self.threshold
            kernel = kernel/np.quantile(kernel,self.quantile)
            self.kernel = kernel

        elif(self.approx=='bfgs'):
            invH = 1./self.kernel
            dg = cg - pg
            dx = cm - pm
            rho = 1./np.sum(dx*dg,axis=1)
            gh = np.sum(dg**2*invH, axis=1)
            kernel = (rho**2*gh+rho)*dx**2 + invH - 2*rho*dx*dg*invH
            kernel = 1./kernel
            self.kernel = np.sqrt(np.mean(kernel**2,axis=0))

        elif(self.approx=='delta'):
            self.dm = (1-self.alpha)*self.dm + self.alpha*np.mean((cm - pm)**2,axis=0)
            self.dg = (1-self.alpha)*self.dg + self.alpha*np.mean((cg - pg)**2,axis=0)
            kernel = self.dg/(self.dm+eps)
            kernel = np.sqrt(kernel)
            kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        elif(self.approx=='grad'):
            self.dg = (1-self.alpha)*np.mean(cg**2,axis=0) + self.alpha*self.dg
            kernel = np.sqrt(self.dg)
            #kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        elif(self.approx=='adam'):
            self.dm = (1-self.alpha)*cg + self.alpha*self.dm
            self.dg = (1-self.beta)*cg**2 + self.beta*self.dg
            kernel = np.mean(self.dg,axis=0)/np.mean(np.abs(self.dm)+eps,axis=0)
            kernel = kernel/np.quantile(kernel,self.quantile)
            kernel[kernel<self.threshold] = self.threshold
            self.kernel = kernel

        return self.kernel

