import pymc3 as pm
import numpy as np
import theano
import theano.tensor as tt
from pyfm2d import fm2d


class LogLikeGrad(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dscalar]

    def __init__(self, model, data, src, rec, mask, sigma, nx=11, ny=11, xmin=-5, ymin=-5, dx=0.1, dy=0.1, gdx=5, gdy=5, sdx=4, sext=4, earth=0):
        self.model = model
        self.data = data
        self.sigma = sigma
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.ymin = ymin
        self.dx = dx
        self.dy = dy
        self.gdx = gdx
        self.gdy = gdy
        self.sdx = sdx
        self.sext = sext
        self.earth = earth
        self.mask = np.ascontiguousarray(mask)

        self.src = src
        self.rec = rec
        # load sources and receivers
        #src = np.loadtxt('sources.dat')
        self.srcx = np.ascontiguousarray(src[:,0])
        self.srcy = np.ascontiguousarray(src[:,1])
        #rec = np.loadtxt('receivers.dat')
        self.recx = np.ascontiguousarray(rec[:,0])
        self.recy = np.ascontiguousarray(rec[:,1])

        # initialise the gradient op
        self.logpgrad = LogGrad(self.model,self.data,self.src,self.rec,self.mask,
                                self.sigma,self.nx,self.ny,self.xmin,self.ymin,
                                self.dx,self.dy,self.gdx,self.gdy,self.sdx,
                                self.sext, self.earth)

    def perform(self, node, inputs, outputs):
        theta, = inputs

        self.time, self.dtdv = self.model(theta,self.srcx,self.srcy,self.recx,self.recy,
                                           self.mask,self.nx,self.ny,self.xmin,self.ymin,
                                           self.dx,self.dy,self.gdx,self.gdy,self.sdx,
                                           self.sext,self.earth)
        # get likelihood
        lglike = np.sum(-(0.5/self.sigma**2)*((self.data-self.time)**2))

        outputs[0][0] = np.array(lglike)

    def grad(self, inputs, g):

        theta, = inputs
        #lggrad = self.dtdv*((self.data-self.time)/self.sigma**2)[:,None]
        #lggrad = self.logpgrad(theta)*((self.data-self.time)/self.sigma**2)[:,None]
        #lggrad = np.sum(lggrad,axis=0)
        lggrad = self.logpgrad(theta)

        print(lggrad)
        return [g[0]*lggrad]

class LogGrad(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, model, data, src, rec, mask, sigma, nx=11, ny=11, xmin=-5, ymin=-5, dx=0.1, dy=0.1, gdx=5, gdy=5, sdx=4, sext=4, earth=0):
        self.model = model
        self.data = data
        self.sigma = sigma
        self.nx = nx
        self.ny = ny
        self.xmin = xmin
        self.ymin = ymin
        self.dx = dx
        self.dy = dy
        self.gdx = gdx
        self.gdy = gdy
        self.sdx = sdx
        self.sext = sext
        self.earth = earth
        self.mask = np.ascontiguousarray(mask)

        self.src = src
        self.rec = rec
        # load sources and receivers
        #src = np.loadtxt('sources.dat')
        self.srcx = np.ascontiguousarray(src[:,0])
        self.srcy = np.ascontiguousarray(src[:,1])
        #rec = np.loadtxt('receivers.dat')
        self.recx = np.ascontiguousarray(rec[:,0])
        self.recy = np.ascontiguousarray(rec[:,1])

    def perform(self, node, inputs, outputs):
        theta, = inputs

        self.time, self.dtdv = self.model(theta,self.srcx,self.srcy,self.recx,self.recy,
                                           self.mask,self.nx,self.ny,self.xmin,self.ymin,
                                           self.dx,self.dy,self.gdx,self.gdy,self.sdx,
                                           self.sext,self.earth)
        # get likelihood
        lggrad = self.dtdv*((self.data-self.time)/self.sigma**2)[:,None]
        grads = np.sum(lggrad,axis=0)


        outputs[0][0] = grads

