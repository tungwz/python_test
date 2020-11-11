# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:50:32 2019

@author: ZongrAx
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt

nx = 1000
dx = 1
x = list(range(0, 1000))

Cxx = np.ones((1000, 1000), dtype=np.float)

for i in range(0, 1000):
    for j in range(i, 1000):
        dij = min(x[j] - x[i], x[i] + nx*dx - x[j])
        Cxx[i][j] = Cxx[i][j] * math.exp(-dij**2 / 20**2)
        Cxx[j][i] = Cxx[i][j]   # 1000x1000

plt.figure(1)
plt.title(r'$C^0_{aa}$')
plt.imshow(Cxx)
plt.colorbar()

#mu = [0 for _ in range(nx)]
f0 = np.random.multivariate_normal(
    np.zeros((nx), dtype=np.float), Cxx)  # true initial state
f0 = np.transpose(f0)

plt.figure(2)
plt.title('Initial value')
plt.plot(f0)
plt.show()

os.system('pause')
c = 1
dt = 1
nt = 2000
ft = f0

e = np.eye((1000), dtype=np.float)  # 1000x1000
d = np.eye(1000, k=-1, dtype=np.float)
G = np.dot(e, (1 - c*dt/dx)) + np.dot(d, (c*dt/dx))
G[0][999] = c * dt / dx  # 1000x1000

# first guess solution
fa0 = f0 + \
    np.transpose(np.random.multivariate_normal(
        np.zeros((nx), dtype=np.float), Cxx))
fa = fa0  # 1000x1
Caa = Cxx  # 1000x1000
Cqq = np.zeros((1000, 1000), dtype=np.float)
# Cqq = Cxx / 10  # 1000x1000

Coo = np.dot(0.1, np.eye(4, dtype=np.float))  # 4x4
M = np.zeros((4, 1000), dtype=np.float)  # M[0~3][0~999] 4x1000
M[0][99] = 1
M[1][349] = 1
M[2][599] = 1
M[3][849] = 1
dt_obs = 1

plt.figure(3)
i = 0
for i in range(2000):
    ft = np.dot(G, ft)  # ft[0~999][0] 1000x1
    ff = np.dot(G, fa)  # 1000x1

    if np.mod(i, dt_obs) == 0:
        d = np.dot(M, ft) + np.transpose(np.random.multivariate_normal(
            np.zeros((4), dtype=np.float), Coo))  # 4x1
        Cff = np.dot(np.dot(G, Caa), np.transpose(G)) + Cqq  # 1000x1000
        P = np.linalg.inv(np.dot(np.dot(M, Cff), np.transpose(M)) + Coo)  # 4x4
        K = np.dot(np.dot(Cff, np.transpose(M)), P)  # 1000x4
        Caa = np.dot((np.eye(1000, dtype=np.float) -
                      np.dot(K, M)), Cff)  # 1000x1000
        fa = ff + np.dot(K, (d - np.dot(M, ff)))  # 1000x1
    else:
        fa = ff

    plt.subplot(211)
    plt.cla()
    plt.plot(fa, c='r', label='fa')
    plt.plot(ft, c='b', label='ft')
    plt.legend(loc="upper right")
    if np.mod(i, dt_obs) == 0:
        plt.plot([99, 349, 599, 849], d, 'd', c='k')

    plt.subplot(212)
    if np.mod(i, dt_obs) == 0:
        plt.cla()
        plt.plot(np.diag(Caa))

    plt.suptitle('t = %2f' % (i+1))
    plt.pause(0.01)

plt.show()
