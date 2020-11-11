import math
import numpy as np
import matplotlib.pyplot as plt

nx = 1000
dx = 1
x = list(range(0, 1000))

Cxx = np.ones((nx, nx), dtype=np.float)
for i in range(0, 1000):
    for j in range(i, 1000):
        dij = min(x[j] - x[i], x[i] + nx*dx - x[j])
        Cxx[i][j] = Cxx[i][j]*math.exp(-dij**2 / 20**2)
        Cxx[j][i] = Cxx[i][j]

plt.title(r'$C^0_{aa}$')
plt.imshow(Cxx)

f0 = np.random.multivariate_normal(np.zeros((nx), dtype=np.float), Cxx)
plt.figure(2)
plt.plot(f0)

c = 1
dt = 1
nt = 1000

ft = f0

G = np.eye((nx), dtype=np.float) * (1-c*dt/dx) + \
    np.eye(1000, k=-1, dtype=np.float) * (c*dt/dx)
G[0][999] = c*dt/dx

nset = 50
faset = np.tile(f0, (1, nset)) + np.transpose(np.random.multivariate_normal(
    np.zeros(nx, dtype=np.float), Cxx))
Cqq = Cxx / 100

Coo = 0.1 * np.eye((4, 4), dtype=np.float)
M = np.zeros((4, nx), dtype=np.float)
M[0][99] = 1
M[1][349] = 1
M[2][599] = 1
M[3][849] = 1

for i in range(1000):
    ft = G * ft

    d = M * ft + \
        np.random.multivariate_normal(np.zeros((1, 4), dtype=np.float), Coo)
    fset = G * faset + \
        np.transpose(np.random.multivariate_normal(
            np.zeros(nx, dtype=np.float), Cqq))
    fmean = np.mean(fset, 2)
    Cff = 1/(nset-1) * (fset - np.tile(fmean, (1, nset))) * \
        np.transpose((fset - np.tile(fmean, (1, nset))))
    K = Cff * np.transpose(M) / (M * Cff * np.transpose(M) + Coo)
    Caa = (np.eye((nx), dtype=np.float) - K * M) * Cff
    dset = np.tile(d, (1, nset)) + np.transpose(
        np.random.multivariate_normal(np.zeros(4, dtype=np.float), Coo))
    faset = fset + K * (dset - M * fset)
