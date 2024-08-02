import numpy as np
import initialisation as init
import optimisation as opt
import matplotlib.pyplot as plt
from utils import plot_form_contour, rescale_all, save_current, save_init


def A_shape(X, Y):
    return np.logical_and(np.abs(X) < 1, np.abs(Y) < 1).astype(float)


def B_shape(X, Y):
    return np.logical_and(X + 0.2 < 0.5 * (Y + 0 / 5), (X + 0.2) ** 2 < 0.1).astype(float)


def C_shape(X, Y):
    return np.logical_and(np.abs(X) < 2, np.abs(Y) < 2).astype(float)


if __name__ == '__main__':
    _, A0, B = init.create_grid_and_functions(A_shape, C_shape, 3, (50, 50))
    alpha = np.pi / 6
    G = np.array([[np.cos(alpha), -np.sin(alpha), -1], [np.sin(alpha), np.cos(alpha), 1]])
    B = B.similarity(G)
    grid, A, B = init.initialise(A0, B, align=False)
    save_init(A)
    GA, G = opt.optimise(grid, A, B, nb_iter=100, descent_step=0.5, heaviside_eps=0.5, sdf_iter=100, deform=False)
    save_current(GA, B)


