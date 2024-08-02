import numpy as np
from utils import *
import optimisation as opt
from scipy.linalg import expm

def scalar(v1, v2):
    return np.einsum('ijkl,ijlk->ij', v1[:, :, np.newaxis, :], v2[:, :, :, np.newaxis])


A0 = load_init()
grid, A, B = load_current(True)
v = np.load('v.npy')

fig = plt.figure()
fig.suptitle('Similarity')

v_drawer = VectorFieldDrawer(grid, fig, 131)
g_drawer = SimDrawer(grid, fig, 132)
c_drawer = ScalarFieldDrawer(grid, fig, 133)


v_drawer.draw(v)

dG = opt.similarity_projection2d_adj(A0, v, 0.5)
G = expm(dG)
g_drawer.draw_on_shape(G, A)
g_grid = g_drawer.sim_grid


c = scalar(v, g_grid)
c_drawer.draw(c)

fig2 = plt.figure()
fig2.suptitle('Sim decomposition')
div_drawer = ScalarFieldDrawer(grid, fig2, 131, 'Divergence')
rot_drawer = ScalarFieldDrawer(grid, fig2, 132, 'Rotation')
t_drawer = ScalarFieldDrawer(grid, fig2, 133, 'Translation')

div = scalar(np.dstack((grid.X, grid.Y)), g_grid)
rot = scalar(np.dstack((-grid.Y, grid.X)), g_grid)
t = scalar(np.ones((grid.shape[0], grid.shape[1], 2)), g_grid)

div_drawer.draw(div)
rot_drawer.draw(rot)
t_drawer.draw(t)

plt.ioff()
plt.show()

