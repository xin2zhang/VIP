from pykernel import pdist, svgd_gradient, pyksd
import numpy as np
import scipy.spatial.distance as sd
import kernel
import time


nparticles = 100
ndim = 500000
eps = 1e-6
batchsize = 100000

x = np.random.random((nparticles,ndim))
w = np.full((ndim,),fill_value=1.,dtype=np.float64)

print("Testing pdist function ...")
s = time.time()
#distance = scipy.spatial.distance.pdist(x,metric='minkowski',p=2,w=w)
distance = sd.pdist(x)
distance = sd.squareform(distance)**2
print('scipy time: '+str(time.time()-s))

s = time.time()
dist = pdist(x,w)
print('pysvgd: '+str(time.time()-s))
print(dist[0,0],dist[80,80])
assert (abs(distance-dist)<eps).all(), 'Test failed'

#bdist = np.zeros((nparticles,nparticles))
bdist = 0
sdist = 0
for i in range(0,ndim,batchsize):
    bdist = bdist + pdist(np.ascontiguousarray(x[:,i:i+batchsize]),w[i:i+batchsize])
    #sdist = sdist + sd.squareform(sd.pdist(x[:,i:i+batchsize]))**2
print(bdist[20,10],dist[20,10])
assert (abs(bdist-dist)<eps).all(), 'Batch test failed'

print("Testing svgd gradient ...")
grad = np.random.random((nparticles,ndim))

sgrad = kernel.svgd_gradient(x, grad)

sgrad2 = svgd_gradient(x, grad, dist, w, kernel='rbf')

assert (abs(sgrad-sgrad2)<1e-8).all()

w = np.random.random((ndim,))

s = time.time()
sgrad = kernel.svgd_gradient(x, grad, kernel='diagonal', hessian=w)
print('scipy time: '+str(time.time()-s))

s = time.time()
dist = pdist(x,w)
sgrad2 = svgd_gradient(x, grad, dist, w, kernel='diagonal')
print('pysvgd: '+str(time.time()-s))

assert (abs(sgrad-sgrad2)<eps).all(), 'Test failed'

bgrad = np.zeros((nparticles,ndim))
for i in range(0,ndim,batchsize):
    bgrad[:,i:i+batchsize] = svgd_gradient(np.ascontiguousarray(x[:,i:i+batchsize]),np.ascontiguousarray(grad[:,i:i+batchsize]),dist,w[i:i+batchsize],kernel='diagonal')
assert (abs(bgrad-sgrad2)<eps).all(), 'Test failed'

print('Testing ksd...')
ksd, step = pyksd(x, grad, w, kernel='diagonal')
print('ksd value: '+str(ksd)+' '+str(step))
