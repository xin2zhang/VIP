import numpy as np
from scipy.spatial.distance import pdist, squareform

def sqr_dist(x, w=None):
    # pairwise square distance
    if(w is None):
        sq_dist = pdist(x)
    else:
        sq_dist = pdist(x,metric='minkowski',p=2,w=w)
    pairwise_dists = squareform(sq_dist)**2
    return pairwise_dists

def rbf_kernel(x, h = -1):

    H = sqr_dist(x)
    if h < 0: # if h < 0, using median trick
        h = np.median(H)
        h = np.sqrt(0.5 * h / np.log(x.shape[0]+1))

    # compute the rbf kernel
    #print("RBF kernel h: "+str(h))
    Kxy = np.exp( -H / h**2 / 2)

    dxkxy = -np.matmul(Kxy, x)
    sumkxy = np.sum(Kxy, axis=1, keepdims=True)
    dxkxy = dxkxy + x * sumkxy
    dxkxy = dxkxy / (h**2)
    return (Kxy, dxkxy)

def poly_kernel(x, subtract_mean=True, e=1e-8):
    if subtract_mean:
        x = x - np.mean(x, axis=0)
    kxy = 1 + np.matmul(x, x.T)
    dxkxy = x * x.shape[0]
    return kxy, dxkxy


def imq_kernel(x, h=-1):
    H = sqr_dist(x)
    if h == -1:
        h = np.median(H)

    kxy = 1. / np.sqrt(1. + H / h)

    dxkxy = .5 * kxy / (1. + H / h)
    dxkxy1 = -np.matmul(dxkxy, x)
    sumkxy = np.sum(dxkxy, axis=1, keepdims=True)
    dxkxy = (dxkxy1 + x * sumkxy) * 2. / h

    return kxy, dxkxy

def hessian_kernel(x, H, h=-1):
    n, d = x.shape
    #diff = x[:,None,:] - x[None,:,:] # n*n*d
    #Qdiff = np.matmul(diff,H)
    #Hdist = np.sum(Qdiff * diff, axis = -1)
    Hdist = pdist(x, 'mahalanobis', VI=H)
    Hdist = squareform(Hdist)**2
    if(h==-1):
        h = np.median(Hdist)
        h = 0.5*h/np.log(n)

    kxy = np.exp(-Hdist/(2.*h))
    #dxkxy = -2*diff*kxy[:,:,None]/h
    #dxkxy = np.sum(dxkxy,axis=1)
    dxkxy = - np.matmul(kxy,x)
    sumkxy = np.sum(kxy,axis=1,keepdims=True)
    dxkxy = dxkxy + x*sumkxy
    dxkxy = dxkxy/h
    return kxy, dxkxy

def diagonal_kernel(x, diag, h=-1):
    ''' x    a 2D array (nparticles,d)
        diag a 1d vector of (d)
    '''
    n, d = x.shape
    #diff = x[:,None,:] - x[None,:,:]
    Hdist = sqr_dist(x, w=diag)
    if(h==-1):
        h = np.median(Hdist)
        h = 0.5*h/np.log(n+1)

    kxy = np.exp(-Hdist/(2.*h))
    #dxkxy = -2*diff*kxy[:,:,None]/h
    #dxkxy = np.sum(dxkxy,axis=1)
    dxkxy = - np.matmul(kxy,x)
    sumkxy = np.sum(kxy,axis=1,keepdims=True)
    dxkxy = dxkxy + x*sumkxy
    dxkxy = dxkxy/h
    return kxy, dxkxy

def svgd_gradient(x, grad, kernel='rbf', hessian=None, ihessian=None, temperature=1., u_kernel=None, **kernel_params):
    assert x.shape[1:] == grad.shape[1:], 'illegal inputs and grads'
    n, d = x.shape

    if u_kernel is not None:
        kxy, dxkxy = u_kernel['kxy'], u_kernel['dxkxy']
        #dxkxy = np.reshape(dxkxy, x.shape)
    else:
        if kernel == 'rbf':
            kxy, dxkxy = rbf_kernel(x, **kernel_params)
            svgd_grad = (np.matmul(kxy, grad) + temperature * dxkxy) / n
        elif kernel == 'poly':
            kxy, dxkxy = poly_kernel(x)
            svgd_grad = (np.matmul(kxy, grad) + temperature * dxkxy) / n
        elif kernel == 'imq':
            kxy, dxkxy = imq_kernel(x, **kernel_params)
            svgd_grad = (np.matmul(kxy, grad) + temperature * dxkxy) / n
        elif kernel == 'hessian':
            kxy, dxkxy = hessian_kernel(x, hessian, **kernel_params)
            sgrad = np.sum(kxy[:,:,None]*grad[None,:,:],axis=1)
            #invH = np.linalg.inv(hessian)
            invH = ihessian
            svgd_grad = (np.matmul(sgrad,invH) + temperature * dxkxy)/n
        elif kernel == 'diagonal':
            kxy, dxkxy = diagonal_kernel(x, hessian, **kernel_params)
            invH = 1./hessian
            sgrad = invH[None,:]*grad[:,:]
            svgd_grad = (np.matmul(kxy,sgrad) + temperature *  dxkxy)/n
        elif kernel == 'separate':
            kxy, dxkxy = rbf_kernel(x, **kernel_params)
            svgd_grad = (np.matmul(kxy,sgrad) + temperature *  dxkxy)/n
            svgd_grad = np.matmul(svgd_grad,hessian)
        elif kernel == 'none':
            kxy = np.eye(x.shape[0])
            dxkxy = np.zeros_like(x)
            svgd_grad = (np.matmul(kxy, grad) + temperature * dxkxy) / n
        else:
            raise NotImplementedError


    return svgd_grad
