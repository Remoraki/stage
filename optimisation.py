from grids import *
import numpy as np
import matplotlib.pyplot as plt
from findiffs import *


def soft_heaviside(x, epsilon=0.1):
    return 0.5*(1 + (2/np.pi)*np.atan(x/epsilon))


def soft_delta(x, epsilon=0.1):
    return 0.5*(2/np.pi) * (1/(1 + (x/epsilon)**2)) / epsilon


# Loading the forms
grid_info = np.load('grid.npy', allow_pickle=True).item()
grid = Grid2D(grid_info['bl'], grid_info['tr'], grid_info['res'])
A = grid.form(np.load('A.npy'))
B = grid.form(np.load('B.npy'))

# Constants
heaviside_eps = 0.5
sdf_iter = 500
descent_step = 0.01

# Symmetrical difference
phi_A = A.get_sdf(sdf_iter)
phi_B = B.get_sdf(sdf_iter)
chi_A = soft_heaviside(-phi_A, heaviside_eps)
chi_B = soft_heaviside(-phi_B, heaviside_eps)
sym_diff = chi_A - chi_B

# Heaviside derivative
delta_phi_A = soft_delta(-phi_A, heaviside_eps)

# Level set gradient
grad_phi_A_x = (fdiff_x(phi_A) - bdiff_x(phi_A)) / (2 * grid.dS[0])
grad_phi_A_y = (fdiff_y(phi_A) - bdiff_y(phi_A)) / (2 * grid.dS[1])
grad_phi_A = np.stack((grad_phi_A_x, grad_phi_A_y), axis=-1)

# Descent direction
step = sym_diff[:, :, np.newaxis] * delta_phi_A[:, :, np.newaxis] * grad_phi_A

# Deformation
GA = A.deform(step, descent_step)
phi_GA = GA.get_sdf(sdf_iter)
chi_GA = soft_heaviside(-phi_GA, heaviside_eps)
sym_diff_G = chi_GA - chi_B


fig0 = plt.figure()
fig0.suptitle('Vector field decomposition')
ax0 = plt.subplot(231, projection='3d')
ax0.plot_surface(grid.X, grid.Y, sym_diff)
ax0.set_title('Soft symmetrical difference')
ax1 = plt.subplot(232, projection='3d')
ax1.plot_surface(grid.X, grid.Y, sym_diff * delta_phi_A)
ax1.set_title('Soft heaviside derivative component')
ax2 = plt.subplot(233, projection='3d')
ax2.plot_surface(grid.X, grid.Y, np.linalg.norm(sym_diff[:, :, np.newaxis] * grad_phi_A, axis=-1))
ax2.set_title('Sdf gradient norm component')
ax3 = plt.subplot(234)
ax3.quiver(grid.X, grid.Y, step[:, :, 0], step[:, :, 1])
ax3.set_title('Vector field')
ax4 = plt.subplot(235, projection='3d')
ax4.plot_surface(grid.X, grid.Y, np.linalg.norm(step, axis=-1))
ax4.set_title('Vector field norm')

grid, [A, GA, B] = rescale_all([A, GA, B])

fig1 = plt.figure()
fig1.suptitle('Shape deformation')
ax5 = plt.subplot(131, projection='3d')
A.plot_surface(ax5, 'Original shape')
ax6 = plt.subplot(132, projection='3d')
GA.plot_surface(ax6, 'Deformed shape')
ax7 = plt.subplot(133, projection='3d')
B.plot_surface(ax7, 'Target shape')


print('Sym diff before deformation : ' + str(np.sum(np.abs(sym_diff))))
print('Sym diff after deformation : ' + str(np.sum(np.abs(sym_diff_G))))
np.save('A', GA.chi)

plt.show()











