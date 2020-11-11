import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fa = np.loadtxt('fa.txt')
ft = np.loadtxt('ft.txt')
d = np.loadtxt('d.txt')
Caa = np.loadtxt('caa.txt')


def update(i):
    line.set_ydata(fa[1000*i:1000*(i + 1)])
    line_t.set_ydata(ft[1000*i:1000*(i + 1)])
    dot.set_ydata(d[4*i:4*(i + 1)])
    caa.set_ydata(Caa[1000*i:1000*(i + 1)])
    suptitle = plt.suptitle('t = %2f' % (i))
    return line, line_t, caa, dot, suptitle


fig = plt.figure(1)

plt.subplot(211)
plt.xlim(0, 1000)
plt.ylim(-5, 5)
line, = plt.plot(fa[0:1000], c='r')
line_t, = plt.plot(ft[0:1000], c='b')
dot, = plt.plot([100, 350, 600, 850], d[0:4], 'o', c='k')

plt.subplot(212)
caa, = plt.plot(Caa[0:1000], c='k')
k = np.max(Caa)
plt.xlim(0, 1000)
plt.ylim(0, k+1)
plt.suptitle('t = 1')

anim = FuncAnimation(fig, update, frames=range(2000))
#anim.save('KF.gif', fps=60)
