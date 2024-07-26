from grids import *
import numpy as np
from utils import load_current, plot_form_surface


def invG(G):
    R = G[0:1, 0:1]
    t = G[0:1, 2]
    return np.hstack((np.transpose(R), -t))


class Similarity2D:

    def __init__(self, G, chain=None):
        if chain is None:
            chain = []
        self.G = G
        self.chain = chain

    def __call__(self, form: GridForm2D):
        f = form
        for g in self.chain:
            f = f.similarity(g)
        return f.similarity(self.G)

    def inv(self, form: GridForm2D):
        f = form
        f = f.similarity(invG(self.G))
        for g in reversed(self.chain):
            f = f.similarity(invG(g))
        return f

    def add(self, G):
        self.chain.append(G)

    def o(self, sim):
        chain = sim.chain + [sim.G] + self.chain
        return Similarity2D(self.G, chain)


if __name__ == '__main__':
    theta = - np.pi / 4
    sigma = 1.1
    t = np.array([-0.2, 0.5])
    R = sigma * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    G = np.hstack((R, t[:, np.newaxis]))
    sim = Similarity2D(G)
    sim2 = sim.o(sim)

    _, A, B = load_current()
    A1 = sim(A)
    A2 = sim(A1)
    A3 = sim2(A)

    _, [A, A1, A2, A3] = rescale_all([A, A1, A2, A3], 0.1,True)
    plot_form_surface(A, 'A')
    plot_form_surface(A1, 'A1')
    plot_form_surface(A2, 'A2')
    plot_form_surface(A3, 'A3')
    plt.show()



# Plotting
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
