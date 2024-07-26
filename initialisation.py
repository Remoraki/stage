import numpy as np

from grids import *
import matplotlib.pyplot as plt
from scipy.linalg import eig
from utils import *

"""
Initialise the shape to align the moments
"""

print('Creating grid and functions')
grid = Grid2D([-2, -2], [2, 2], [100, 100])
B_chi = np.logical_and(grid.X + 0.2 < 0.5 * (grid.Y + 0 / 5), (grid.X + 0.2) ** 2 < 0.1).astype(float)
A_chi = np.logical_and(np.abs(grid.X) < 0.4, np.abs(grid.Y) < 1.6).astype(float)
B = grid.form(B_chi)
B = B.rotate_from_angle(np.pi / 6)
A = grid.form(A_chi)


print('Calculating moments')
area_B, barycenter_B, cov_B = B.get_area(), B.get_barycenter(), B.get_covariance_matrix()
area_A, barycenter_A, cov_A = A.get_area(), A.get_barycenter(), A.get_covariance_matrix()
eig_values_B, eig_vectors_B = eig(cov_B)
eig_values_A, eig_vectors_A = eig(cov_A)
ei_values_A, eig_values_B = np.real(eig_values_A), np.real(eig_values_B)

print('Calculating transformation')
t = barycenter_B - barycenter_A
sigma = np.sqrt(np.sum(eig_values_B))
R = eig_vectors_B
G = np.hstack((sigma*R, t[:, np.newaxis]))
GA = A.similarity(G)

print('Plotting')
fig0 = plt.figure()
fig0.suptitle('Target shape')
ax0 = plt.subplot(121, projection='3d')
ax1 = plt.subplot(122, projection='3d')
fig1 = plt.figure()
fig1.suptitle('Original shape')
ax2 = plt.subplot(121, projection='3d')
ax3 = plt.subplot(122, projection='3d')
fig2 = plt.figure()
fig2.suptitle('Initialised shape')
ax4 = plt.subplot(121, projection='3d')
ax5 = plt.subplot(122, projection='3d')

(_, [A, B, GA]) = rescale_all([A, B, GA], 0.2, True)


B.plot_surface(ax0)
B.plot_sdf(ax1, 500)
A.plot_surface(ax2)
A.plot_sdf(ax3, 500)
GA.plot_surface(ax4)
GA.plot_sdf(ax5, 500)


grid, [GA, B] = rescale_all([GA, B], 0.2, True)
save(grid, GA, B)

plt.show()

