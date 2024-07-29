import numpy as np
import initialisation as init
import optimisation as opt
import matplotlib.pyplot as plt
from utils import plot_form_contour, plot_form_surface, rescale_all


def A_shape(X,Y):
    return np.logical_and(np.abs(X) < 0.4, np.abs(Y) < 1.6).astype(float)


def B_shape(X,Y):
    return np.logical_and(X + 0.2 < 0.5 * (Y + 0 / 5), (X + 0.2) ** 2 < 0.1).astype(float)


if __name__ == '__main__':
    grid, A0, B = init.create_grid_and_functions(A_shape, B_shape, 3, (100, 100))
    alpha = np.pi / 6
    G = np.array([[np.cos(alpha), -np.sin(alpha), 0.5], [np.sin(alpha), np.cos(alpha), 0]])
    B = B.similarity(G)
    grid, A, B = init.initialise(A0, B)
    #grid, [A, B] = rescale_all([A0, B])
    GA = opt.optimise(grid, A, B, 100, descent_step=2)


