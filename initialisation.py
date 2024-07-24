import numpy as np

from grids import *
import matplotlib.pyplot as plt
from scipy.linalg import eig

print('Creating grid and functions')
grid = Grid2D([-2, -2], [2, 2], [100, 100])
B = np.logical_and(grid.X + 0.2 < 0.5 * (grid.Y + 0 / 5), (grid.X + 0.2) ** 2 < 0.1).astype(float)
A = np.logical_and(np.abs(grid.X) < 0.4, np.abs(grid.Y) < 1.6).astype(float)
B = grid.form(B)
B = grid.rotate(B, np.pi / 6)
A = grid.form(A)


print('Calculating moments')
area_B, barycenter_B, cov_B = B.get_area(), B.get_barycenter(), B.get_covariance_matrix()
area_A, barycenter_A, cov_A = A.get_area(), A.get_barycenter(), A.get_covariance_matrix()

eig_values_B, eig_vectors_B = eig(cov_B)
eig_values_A, eig_vectors_A = eig(cov_A)

print('Calculating transformation')
t = barycenter_B - barycenter_A
sigma = np.sqrt(np.sum(eig_values_B))
R = eig_vectors_B
G = np.hstack((sigma*R, t[:, np.newaxis]))

coords = A.get_coords(True)
transformed_coords = np.matmul(G, coords)
GA = grid.form_interpolation(transformed_coords)

print('Plotting')
ax0 = plt.subplot(321, projection='3d')
ax1 = plt.subplot(322, projection='3d')
ax2 = plt.subplot(323, projection='3d')
ax3 = plt.subplot(324, projection='3d')
ax4 = plt.subplot(325, projection='3d')
ax5 = plt.subplot(326, projection='3d')

(grid, [A, B, GA]) = rescale_all([A, B, GA])

B.plot_surface(ax0)
B.plot_sdf(ax1, 500)
A.plot_surface(ax2)
A.plot_sdf(ax3, 500)
GA.plot_surface(ax4)
GA.plot_sdf(ax5)

plt.show()

