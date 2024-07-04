import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import *
from shapes import *


# point set definition
n = 10000

generation = 'sphere'

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
if generation == 'sphere':
    points = generate_sphere_points(1, n)
    x = [points[k][0] for k in range(len(points))]
    y = [points[k][1] for k in range(len(points))]
    z = [points[k][2] for k in range(len(points))]
# color scheme for plotting
colors = np.linspace(0, 1, n)

# 3D scatter plot
fig = plt.figure()
plt.title('3D set of points')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=colors)
plt.show()

# camera definition
look_at = np.array([2, 0, 0])
pos = np.array([0, -3, 0])
up = [0, 0, 1]
cam = get_camera_from_vector(pos, look_at, up)

for i in range(n):
    cam.see_point([x[i], y[i], z[i]])

fig, ax = cam.get_plot()
cam.render(ax)
cam.wipe()
plt.show()



