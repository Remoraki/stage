import numpy as np
from tqdm import tqdm

from Levelsets import *


shape = np.transpose(np.load('C:/2A/Stage/sfs_voxels.npy'))
size = 0.5
res = (2**6)
X, Y, Z = np.meshgrid(np.linspace(-size, size, res), np.linspace(-size, size, res), np.linspace(-size, size, res))
coords = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
chi = np.zeros(coords.shape)

distances = np.linalg.norm(coords[:, None, :] - shape[None, :, :], axis=-1)
args = np.argmin(distances, axis=0)
chi[args] = 1

print(np.sum(chi))
