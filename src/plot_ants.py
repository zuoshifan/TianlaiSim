import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# def randrange(n, vmin, vmax):
#     return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# n = 100
# for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zl, zh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)

antpos = np.array([
         [    0.,    0.,    0.],
         [-100.67, 138.14,-198.28],
         [ -68.62, 381.81,-133.88],
         [  59.4, 465.3, 120.0],
    ])
x = antpos.T[0]
y = antpos.T[1]
z = antpos.T[2]
ax.scatter(x,y,z)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()