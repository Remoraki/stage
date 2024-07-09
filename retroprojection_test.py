import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from camera import *
from tqdm import tqdm
from SFS import *

size = 1
sfs = ShapeFromSilhouette((0, 0, 0), size, 50, 0)

# definition of the shape
s_X, s_Y, s_Z = [], [], []
shape = 'square'

if shape == 'square':
    res = 10
    sq_size = np.random.rand()
    sq_x = np.random.rand() - 0.5
    sq_y = np.random.rand() - 0.5
    sq_z = np.random.rand() - 0.5
    sq_X = np.linspace(sq_x, sq_x + sq_size, res)
    sq_Y = np.linspace(sq_y, sq_y + sq_size, res)
    sq_Z = np.linspace(sq_z, sq_z + sq_size, res)
    s_X, s_Y, s_Z = np.meshgrid(sq_X, sq_Y, sq_Z)

# plot of the 3d shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(s_X, s_Y, s_Z)
ax.set_xlim([-2*size, 2*size])
ax.set_ylim([-2*size, 2*size])
ax.set_zlim([-2*size, 2*size])
ax.set_title('3D shape')


# definition of the cameras
cam = get_camera_from_vector('cam1', (0, -1, 0), (0, 0, 0), (0, 0, 1))
cam2 = get_camera_from_vector('cam2', (-1, 0, 0), (0, 0, 0), (0, 0, 1))
cam3 = get_camera_from_vector('cam3', (1, -2, 1), (0, 0, 1), (0, 0, 1))
cameras = [cam, cam2, cam3]

# showing the cameras
ax.scatter([0, -1, 1], [-1, 0, -2], [0, 0, 1], c='r', marker='x')

plt.show()

# camera rendering of the shape
for m in zip(s_X.flatten(), s_Y.flatten(), s_Z.flatten()):
    for camera in cameras:
        camera.place_point(m)

for camera in cameras:
    camera.render()

plt.show()

# voxel projection calculation
shape = np.array(sfs.reconstruct_from_cameras(cameras))

# plotting the reconstructed shape
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(shape[:, 0], shape[:, 1], shape[:, 2])
ax.set_xlim([-size, size])
ax.set_ylim([-size, size])
ax.set_zlim([-size, size])
ax.set_title('shape from silhouette')
plt.show()

