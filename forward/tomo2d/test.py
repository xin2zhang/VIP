# test python wrapper for fortran fmm 2d code

import numpy as np
from pyfm2d import fm2d

srcx = np.random.rand(8)
srcy = np.random.rand(8)
mask = np.zeros((2,64),dtype=np.int32)
for i in range(8):
    for j in range(8):
        if(j>i):
             mask[0,i*8+j] = 1
             mask[1,i*8+j] = i*8 + j + 1
nx = 11
ny = 11
xmin = 0
ymin = 0
dx = 0.1
dy = 0.1
gdx = 2
gdy = 2
sdx = 4
sext =4
vel = np.random.rand(121)

print(srcx)
print(srcy)
print(vel)

recx = srcx
recy = srcy
time, dtdv = fm2d(vel,srcx,srcy,recx,recy,mask,nx,ny,xmin,ymin,dx,dy,gdx,gdy,sdx,sext)

w = np.where(mask[0,:]>0.001)[0]
print(time[w])

print(dtdv[w,:])
