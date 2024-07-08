import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_sphere_points(radius, num_points):
    points = []

    for _ in range(num_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, 2 * np.pi)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)

        points.append([x, y, z])

    return np.array(points)


