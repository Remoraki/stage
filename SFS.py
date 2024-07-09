import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from camera import *


class ShapeFromSilhouette:
    def __init__(self, center, size, n, max_recursion=0):
        """

        :param center: the center of the 3D grid
        :param size: the half-size of the grid
        :param n: the number of point per axis
        :param max_recursion: the maximum recursion depth allowed for dynamic resolution reconstruction
        """
        x = np.linspace(center[0] - size, center[0] + size, n)
        y = np.linspace(center[1] - size, center[1] + size, n)
        z = np.linspace(center[2] - size, center[2] + size, n)

        self.size = size
        self.n = n
        self.VX, self.VY, self.VZ = np.meshgrid(x, y, z)
        self.reconstruction = np.array([[[0] * n] * n] * n)
        self.sfs = []
        self.max_recursion = max_recursion
        self.center = center

    def reconstruct_from_cameras(self, cameras):
        self.reconstruction = np.array([[[0] * self.n] * self.n] * self.n)
        self.sfs = []
        for index in tqdm(np.ndindex(self.VX.shape), 'reconstructing shape'):
            m = (self.VX[index], self.VY[index], self.VZ[index])
            for camera in cameras:
                p = camera.projection(m)
                if p is not None:
                    r = rasterize(p)
                    self.reconstruction[index] += camera.get_pixel_value(r)
            if self.reconstruction[index] >= len(cameras):
                if self.max_recursion <= 0:
                    self.sfs.append(m)
                else:
                    sub_sfs = ShapeFromSilhouette(
                        m, self.size / self.n, self.n, self.max_recursion - 1)
                    sub_shape = sub_sfs.reconstruct_from_cameras(cameras)
                    self.sfs = self.sfs + sub_shape
        return self.sfs
