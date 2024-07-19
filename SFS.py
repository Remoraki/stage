from time import time

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from camera import *


def get_voxel_vertices(x, y, z, d):
    vertices = [
        [x+d, y+d, z+d],
        [x+d, y+d, z-d],
        [x+d, y-d, z+d],
        [x+d, y-d, z-d],
        [x-d, y+d, z+d],
        [x-d, y+d, z-d],
        [x-d, y-d, z+d],
        [x-d, y-d, z-d],
    ]
    return np.transpose(np.array(vertices))


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

        self.voxel_size = 2*size / n
        self.size = size
        self.n = n
        self.VX, self.VY, self.VZ = np.meshgrid(x, y, z)
        self.points = np.vstack([self.VX.ravel(), self.VY.ravel(), self.VZ.ravel()])
        self.sfs = np.empty((3, 0))
        self.max_recursion = max_recursion
        self.center = center

    def reconstruct_from_cameras_matrix(self, cameras):
        """
        Reconstruct a 3D voxel silhouette from different cameras
        :param cameras: A list of cameras
        :return: The silhouette voxel centers
        """
        print('Finding matching voxels')
        points = self.get_voxel_centers(cameras)
        # reconstructing shape inside the voxel grid
        reconstruction = np.array([0] * points.shape[1])
        self.sfs = np.empty((3, 0))
        # scanning all the cameras
        for camera in tqdm(cameras, 'reconstructing shape'):
            # projecting the centers of the voxels on the screen
            P, mask = camera.projection_matrix(points)
            # incrementing each voxel that lands on a binary mask
            binary_mask = [camera.get_pixel_value(P[:, k]) for k in range(P.shape[1])]
            mask = mask[np.where(binary_mask)]
            reconstruction[mask] += 1
        # keep only the voxels that landed on all the masks
        self.sfs = points[:, np.where(reconstruction == len(cameras))[0]]
        return self.sfs

    def get_voxel_centers(self, cameras):
        """
        Scan recursively the grid to determine which areas might contain the silhouette. This prevents computing the
        projection of many useless voxels.
        :param cameras: The lit of cameras.
        :return: A list of the voxel centers to compute.
        """
        if self.max_recursion <= 0:
            return self.points
        else:
            centers = np.empty((3, 0))
            # look for a smaller resolution when useful
            for k in range(self.points.shape[1]):
                point = self.points[:, k]
                voxel_vertices = get_voxel_vertices(point[0], point[1], point[2], self.voxel_size)
                # we explore only voxel that project on every camera
                count = 0
                inside_count = 0
                for camera in cameras:
                    voxel_projections, _ = camera.projection_matrix(voxel_vertices, False)
                    interpolation = camera.scan_points(voxel_projections[0, :], voxel_projections[1, :])

                    #print('Interp : ' + str(interpolation))
                    #plt.figure()
                    #plt.title(camera.name)
                    #plt.imshow(camera.pixel_values)
                    #plt.scatter(voxel_projections[0, :], voxel_projections[1, :])
                    #plt.show()

                    if interpolation == 0:
                        count += 1
                    elif interpolation == 1:
                        count += 1
                        inside_count += 1
                    else:
                        break
                if count == len(cameras) and inside_count != len(cameras):
                    sub_sfs = ShapeFromSilhouette(point, self.voxel_size / 2, self.n, self.max_recursion - 1)
                    centers = np.concatenate((centers, sub_sfs.get_voxel_centers(cameras)), axis=1)
        return centers


