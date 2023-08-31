import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import aco2d
import matplotlib.pyplot as plt
import multiprocessing as mp
#from torch.multiprocessing import Pool, Process
import time

def gradient(vel):
    # vel = vel.astype(np.float64)
    data = aco2d.forward(vel, dim = 1*1001*200, verbose = 0, paramfile='./input_param.txt')

    return data

ns = 1

vel = np.full((200*100,), 2000.0)
vel[50*100:] = 2500.0
vel[100*100:] = 3000.0

rec = aco2d.forward(vel, dim = ns*1001*200, verbose = 0, paramfile='./input_param.txt')

plt.figure()
# plt.imshow(rec.reshape(-1, 200))
plt.imshow(rec.reshape(ns, 1001, 200).transpose(1, 0, 2).reshape(1001, -1), clim=(-0.3, 0.3))
plt.show()

#np.savetxt('./input/shotrecord.txt', rec)
# # print(rec.shape)

# record = np.loadtxt('./input/shotrecord.txt', dtype=np.float32)

vel = np.full((200*100,), 2000.0)
vel[50*100:] = 2300.0
vel[100*100:] = 2700.0
# vel = np.repeat(vel[None, :], 12, axis=0)
# print(vel.shape)
start = time.time()

# pool = Pool(processes = 6)
# results = pool.map(gradient, [vel[i,:].squeeze() for i in range(6)])
# pool.close()
# pool.join()
data, grad = aco2d.fwi(vel, rec, verbose = 0, paramfile='./input_param.txt')

end = time.time()
print('The test time is: ', end-start)

plt.figure()
plt.imshow(data.reshape(ns, 1001, 200).transpose(1, 0, 2).reshape(1001, -1), clim=(-0.3, 0.3))
plt.show()

plt.figure()
plt.imshow(grad.reshape(-1, 200))
plt.show()
