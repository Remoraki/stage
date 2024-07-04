import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera import *

# point set definition
n = 100

generation = 'parallel'

# random generation
if generation == 'random':
    x = 10 * (np.random.rand(n) - 0.5)
    y = 10 * (np.random.rand(n) - 0.5)
    z = 10 * (np.random.rand(n) - 0.5)
# parallel generation
if generation == 'parallel':
    x = (n // 2) * [-1] + (n // 2) * [1]
    y = np.concatenate((np.linspace(-1, 1, n // 2), np.linspace(-1, 1, n // 2)))
    z = np.concatenate((np.linspace(-1, 1, n // 2), np.linspace(-1, 1, n // 2)))
# color scheme for plotting
colors = np.linspace(0, 1, n)

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors)
plt.show()

# camera definition
look_at = np.array([0, 0, 0])
pos = np.array([0, 0, 1])
up = [0, -1, 0]
cam = get_camera_from_vector(pos, look_at, up, 0.005, 0, 0)

# camera projection
u = []
v = []
for i in range(n):
    p = cam.projection([x[i], y[i], z[i]])
    if p is not None:
        u.append(p[0])
        v.append(p[1])

# 2D scatter plot
plt.figure()
colors = np.linspace(0, 1, len(u))
plt.scatter(u, v, c=colors)
plt.xlim([-0.01, 0.01])
plt.ylim([-0.01, 0.01])
plt.show()



