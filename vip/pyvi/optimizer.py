import numpy as np

class optimizer():
    '''
    Class optimizer()
        A class that implements different optimizing algorithms, including 'sgd', 'adagrad', 'adadelta' and 'adam'
    '''
    def __init__(self, shape, lnprob, method='sgd', alpha=0.9, beta=0.99, eps = 1e-6):
        '''
        shape: shape of the variable
        lnprob: log of the probability density function, usually negtive misfit function
        method: optimizing method, including 'sgd','adagrad','adadelta' and 'adam'
        alpha, beta: hyperparameters for optimizer
        eps: a small value to avoid zero dividing
        '''

        self.shape = shape
        self.lnprob = lnprob
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.eps = eps

        self.num = np.zeros(shape)
        self.den = np.zeros(shape)

    def sgd(self, theta, step):
        '''
        Stochastic gradient descent with momentum, set alpha=0 to exlude momentum
        Input
            theta: the current variable
            step: stepsize for current iteration
        Return
            update: update of theta
            loss: loss value for current theta
        '''

        loss, grad, mask = self.lnprob(theta)
        self.num[mask] = 0
        self.num = self.alpha*self.num + step*grad
        update = self.num

        return update, loss

    def adagrad(self, theta, step):
        '''
        Adagrad algorithm, alpha is the hyperparameter that controls decaying rate
        Input
            theta: the current variable
            step: stepsize for current iteration
        Return
            update: update of theta
            loss: loss value for current theta
        '''

        loss, grad, _ = self.lnprob(theta)
        self.den = self.alpha * self.den + (1-self.alpha) * grad**2
        update = step * np.divide(grad, self.eps+np.sqrt(self.den))

        return update, loss

    def adadelta(self, theta, step):
        '''
        Adadelta algorithm, alpha is the hyperparameter that controls decaying rate
        Input
            theta: the current variable
            step: stepsize for current iteration
        Return
            update: update of theta
            loss: loss value for current theta
        '''

        loss, grad, _ = self.lnprob(theta)
        self.den = self.alpha * self.den + (1-self.alpha) * grad**2
        update = step * np.divide(np.sqrt(self.num)+self.eps,np.sqrt(self.den)+self.eps)
        update = update * grad
        self.num = self.alpha * self.num + (1-self.alpha) * update**2

        return update, loss

    def adam(self, theta, step, itr):
        '''
        Adam algorithm, alpha and beta are the hyperparameters that controls decaying rate
        for grad and grad**2 respectively
        Input
            theta: the current variable
            step: stepsize for current iteration
            itr: the current iteration
        Return
            update: update of theta
            loss: loss value for current theta
        '''

        loss, grad, _ = self.lnprob(theta)
        self.num = self.alpha * self.num + (1-self.alpha) * grad
        self.den = self.beta * self.den + (1-self.beta) * grad**2

        m = self.num/(1-self.alpha**(itr))
        v = self.den/(1-self.beta**(itr))

        update = step * np.divide(m, np.sqrt(v)+self.eps)

        return update, loss

    def optim(self, x0, n_iter=1000, stepsize=1e-3, gamma=1.0, decay_step=1):
        '''
        Run the optimization
        Input
            x0: initial value, must have the shape as defined in the constructor
            n_iter: the number of iterations
            stepsize: learning rate / step size
            gamma: decaying rate for stepsize
            decay_step: the number of steps to decay the learning rate
        Return
            theta: optimized value for the variable
            losses: loss value for all iterations
        '''

        losses = np.zeros((n_iter,x0.shape[0]))
        theta = x0

        step = np.full((n_iter,),stepsize)
        for i in range(1,n_iter):
            if(i%decay_step == 0):
                step[i] = stepsize * gamma
            else:
                step[i] = step[i-1]

        for i in range(n_iter):
            print(f'Iteration: {i}')
            if(self.method=='sgd'):
                update, loss = self.sgd(theta,step[i])
            elif(self.method=='adagrad'):
                update, loss = self.adagrad(theta,step[i])
            elif(self.method=='adadelta'):
                update, loss = self.adadelta(theta,step[i])
            elif(self.method=='adam'):
                update, loss = self.adam(theta,step[i],i)
            else:
                print('Not supported optimizer')

            theta += update
            losses[i,:] = loss

        return theta, losses

