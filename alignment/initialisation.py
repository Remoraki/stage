import numpy as np

from grids import *
import matplotlib.pyplot as plt
from scipy.linalg import eig
from utils import *

"""
Script that initialise a model shape and a target shape, and proceed to its alignment via the moments.
"""


def create_grid_and_functions(AFunc, BFunc, grid_size=1, res=(100, 100)):
    """
    Create a grid and the model and target shapes.
    :param AFunc: The characteristic function of A : (X,Y) -> Chi
    :param BFunc: The characteristic function of B : (X,Y) -> Chi
    :param grid_size: The size of the grid
    :param res: The resolution of the grid
    :return: grid:Grid2D, A:GridForm2D, B:GridForm2D
    """
    grid = Grid2D([-grid_size, -grid_size], [grid_size, grid_size], res)
    A_chi = AFunc(grid)
    B_chi = BFunc(grid)
    A = grid.form(A_chi)
    B = grid.form(B_chi)
    return grid, A, B


def create_grids(AGrid, BGrid, grid_size=1, res=100):
    """
    Create a grid and the corresponding functions
    :param AGrid: Grid shape characteristic function
    :param BGrid: Grid shape characteristic function
    :param grid_size: The size of the grid
    :param res: The resolution of the grid (should be the shape of AGrid and BGrid)
    :return:
    """
    grid = Grid2D([-grid_size, -grid_size], [grid_size, grid_size], [res, res])
    A = grid.form(AGrid)
    B = grid.form(BGrid)
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


def initialise(A0: GridForm2D, B: GridForm2D, align=True):
    """
    Initialise shapes
    :param A0:
    :param B:
    :param align: True if the moments of the shape should be aligned
    :return:
    """
    if align:
        G = initial_transformation(A0, B)
        A = A0.similarity(G)
    else:
        A = A0
    grid, [A, B] = rescale_all([A, B], 0.1, False)
    return grid, A, B
