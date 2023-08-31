import numpy as np
import pytrans

ns = 20
n = 401*401*181

x = np.random.rand(ns,n)
lbnd = np.full((n,),fill_value=-0.1)
ubnd = np.full((n,),fill_value=1.1)

#y = np.copy(x)
print(np.mean(x))
pytrans.pytrans(x,lbnd,ubnd)
print(np.mean(x))
pytrans.pyinv_trans(x,lbnd,ubnd)
print(np.mean(x))
#print(np.max(x-y),np.min(x-y))

pytrans.pytrans(x,lbnd,ubnd)

grad = np.random.rand(ns,n)
grad, mask = pytrans.pytrans_grad(grad,x,lbnd,ubnd,rmask=1)
