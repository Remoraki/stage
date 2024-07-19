import os
from time import time

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from tqdm import tqdm

from SFS import ShapeFromSilhouette
from camera import *

# images loading
folder_path = 'C:/2A/Stage/buddha/mask'
images = []
i = 0
for im_name in tqdm(os.listdir(folder_path), 'loading images'):
    image = np.array(Image.open(os.path.join(folder_path, im_name)))
    images.append(image)
    i += 1
nb_of_images = len(images)

# camera data loading
folder_path = 'C:/2A/Stage/buddha'
K = np.load(folder_path + '/K.npy')
R = np.load(folder_path + '/R.npy')
T = np.load(folder_path + '/T.npy')
cameras = []
for i in tqdm(range(nb_of_images), 'creating cameras'):
    cameras.append(Camera('cam ' + str(i), R[:, :, i], T[:, i], K, 1600, 1600))
    cameras[i].pixel_values = images[i]


# reconstruction
size = 0.5
sfs = ShapeFromSilhouette((0, 0, 0), size, 2, 6)
t1 = time()
shape = np.array(sfs.reconstruct_from_cameras_matrix(cameras))
np.save('C:/2A/Stage/sfs_voxels.npy', shape)
t2 = time()
print('Execution time : ' + str(t2 - t1))
print('Number of points : ' + str(shape.shape[1]))

# plotting the reconstructed shape
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(shape[0, :], shape[1, :], shape[2, :])
ax.set_xlim([-size, size])
ax.set_ylim([-size, size])
ax.set_zlim([-size, size])
ax.set_title('shape from silhouette')
plt.show()
