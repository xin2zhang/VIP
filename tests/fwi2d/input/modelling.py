import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import vip.fiw2d.aco2d as aco2d

true_model = np.loadtxt('marmousi_small.txt')
nz = true_model.shape[0]; nx = true_model.shape[1]
print(true_model.shape)
vel = true_model.astype(np.float64).flatten()

nt = 2501
nr = 200
ns = 10
rec = aco2d.forward(vel, dim = ns*nt*nr, verbose = 0, paramfile='./input_params.txt')
rec = rec + 0.1*np.random.normal(size=rec.shape).astype(np.float32)

#np.save('waveforms.npy',rec)
