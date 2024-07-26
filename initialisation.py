import numpy as np

from grids import *
import matplotlib.pyplot as plt
from scipy.linalg import eig
from utils import *

"""
Script that initialise a model shape and a target shape, and proceed to its alignment via the moments.
"""


def create_grid_and_functions(res=100):
    """
    Create a grid and the model and target shapes. Not a generic function, only here to initialise shapes
    :param res: The resolution of the grid
    :return: grid:Grid2D, A:GridForm2D, B:GridForm2D
    """
    grid = Grid2D([-2, -2], [2, 2], [res, res])
    B_chi = np.logical_and(grid.X + 0.2 < 0.5 * (grid.Y + 0 / 5), (grid.X + 0.2) ** 2 < 0.1).astype(float)
    A_chi = np.logical_and(np.abs(grid.X) < 0.4, np.abs(grid.Y) < 1.6).astype(float)
    B = grid.form(B_chi)
    alpha = np.pi/6
    G = np.array([[np.cos(alpha), -np.sin(alpha), 0.5], [np.sin(alpha), np.cos(alpha), 0]])
    B = B.similarity(G)
    A = grid.form(A_chi)
    return grid, A, B


def moments(form: GridForm2D):
    """
    Calculate the moments of a form
    :param form: GridForm2D
    :return: area: float, barycenter:(2), covariance matrix: (2,2)
    """
    return form.get_area(), form.get_barycenter(), form.get_covariance_matrix()


def initial_transformation(A: GridForm2D, B: GridForm2D):
    """
    Calculate a approximate initial transformation with the shape moments
    :param A: The shape to align: GridForm2D
    :param B: The shape to be aligned with: GridForm2D
    :return: A similarity transformation matrix (R,T): (2,3)
    """
    area_A, barycenter_A, cov_A = moments(A)
    area_B, barycenter_B, cov_B = moments(B)
    eig_values_B, eig_vectors_B = eig(cov_B)
    eig_values_A, eig_vectors_A = eig(cov_A)
    ei_values_A, eig_values_B = np.real(eig_values_A), np.real(eig_values_B)
    t = barycenter_B - barycenter_A
    sigma = np.sqrt(np.sum(eig_values_B))
    R = eig_vectors_B
    GAB = np.hstack((sigma * R, t[:, np.newaxis]))
    return GAB # It's a me, GAB


if __name__ == '__main__':
    # Init
    grid, A0, B = create_grid_and_functions(res=50)
    G = initial_transformation(A0, B)
    A = A0.similarity(G)
    # Save
    save_init(A0)
    save_current(A, B)
    # Plot
    plot_form_contour(A0, 'A0')
    plot_form_contour(A, 'A1')
    plot_form_contour(B, 'B')
    plt.show()

