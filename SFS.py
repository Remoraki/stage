import numpy as np
from tqdm import tqdm


class ShapeFromSilhouette:
    def __init__(self, size, n, max_recursion=1):
        x = np.linspace(-size, size, n)
        y = np.linspace(-size, size, n)
        z = np.linspace(-size, size, n)

        self.size = size
        self.n = n
        self.VX, self.VY, self.VZ = np.meshgrid(x, y, z)
        self.reconstruction = np.array([[[0] * n] * n] * n)
        self.shape = []
        self.max_recursion = max_recursion

    def reconstruct_from_cameras(self, cameras):
        self.reconstruction = np.array([[[0] * self.n] * self.n] * self.n)
        self.shape = []
        for index in tqdm(np.ndindex(self.VX.shape), 'reconstructing shape'):
            m = (self.VX[index], self.VY[index], self.VZ[index])
            for camera in cameras:
                p = camera.projection(m)
                if p is not None:
                    r = camera.rasterize(p)
                    self.reconstruction[index] += camera.get_pixel_value(r)
            if self.reconstruction[index] >= len(cameras):
                self.shape.append(m)
        self.shape = np.array(self.shape)
