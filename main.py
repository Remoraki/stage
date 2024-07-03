import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from camera import Camera

n = 10

x = np.random.rand(n)*10 - 0.5
y = np.random.rand(n)*10 - 0.5
z = np.random.rand(n)*10 - 0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
plt.show()


lookat = np.array([0, 0, 0])
pos = np.array([-1, 0, 0])
forward = lookat - pos
forward = forward / np.linalg.norm(forward)

t = [1, 0, 0]
rot = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

cam = Camera(1, rot, t, 0, 0)
u = n*[0]
v = n*[0]

for i in range(n):
    p = cam.projection([x[i], y[i], z[i]])
    u[i] = p[0]
    v[i] = p[1]

plt.figure()
plt.scatter(u, v)
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.show()



