from grids import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from grids import *
from typing import List


def save_init(A0: GridForm2D):
    """
    Save the initial model shape and the grid on which it is defined
    :param A0: The initial model shape: GridForm2D
    :return:
    """
    grid = A0.grid
    grid_dic = {'res': grid.resolution, 'bl': grid.bottom_left, 'tr': grid.top_right}
    np.save('A0', A0.chi)
    np.save('grid', grid_dic, True)


def save_current(A: GridForm2D, B: GridForm2D = None, border_width=0.2, square_output=True):
    """
    Save the current state of the model, the target shape and the grid. B is optional and should not be given as
    argument unless A and B are not defined on the same grid anymore, or if you need to rescale them.
    :param A: The current model shape: GridForm2D
    :param B: The target shape: GridForm2D
    :param border_width: The width of the border when rescaling
    :param square_output: Should the rescaling be squared
    :return:
    """
    if B is not None:
        grid, [A, B] = rescale_all([A, B], border_width, square_output)
        np.save('B', B.chi)
        grid_dic = {'res': grid.resolution, 'bl': grid.bottom_left, 'tr': grid.top_right}
        np.save('grid', grid_dic, True)
    np.save('A', A.chi)


def load_current(load_all=False):
    """
    Load the current state, with the model shape, the target shape, and their common grid if specified
    :param load_all: Do you need to all target shape and common grid as well
    :return: If load_all: (grid, A, B), else:(A)
    """
    grid_info = np.load('grid.npy', allow_pickle=True).item()
    grid = Grid2D(grid_info['bl'], grid_info['tr'], grid_info['res'])
    A = grid.form(np.load('A.npy'))
    if load_all:
        B = grid.form(np.load('B.npy'))
        return grid, A, B
    else:
        return A


def plot_form_surface(form: GridForm2D, title='form', scaled=False, squared_scale=True, fig=None):
    """
    Plot the form with a surface plot
    :param form: GridForm2D
    :param title: The title of the plot
    :param scaled: Should the plot be scaled to the form or respect the form grid
    :param squared_scale: In case of rescaling, should it be squared
    :return:
    """
    if fig is None:
        fig = plt.figure()
    fig.suptitle(title)
    ax0 = plt.subplot(121, projection='3d')
    ax1 = plt.subplot(122, projection='3d')
    if scaled:
        f, _ = form.rescale(squared_scale)
    else:
        f = form
    f.plot_surface(ax0, title='Chi')
    f.plot_sdf_surface(ax1, title='Phi')


def plot_form_contour(form: GridForm2D, title='form', color='red', scaled=False, squared_scale=True, fig=None):
    """
    Plot the form with a contour plot
    :param form: GridForm2D
    :param title: The title of the plot
    :param color: The color of the contour
    :param scaled: Should the plot be scaled to the form or respect the form grid
    :param squared_scale: In case of rescaling, should it be squared
    :return:
    """
    if fig is None:
        fig = plt.figure()
    fig.suptitle(title)
    ax = plt.subplot(121)
    ax2 = plt.subplot(122)
    if scaled:
        f, _ = form.rescale(squared_scale)
    else:
        f = form
    f.plot_contour(ax, color=color)
    f.plot_sdf_im(ax2)
    ax2.invert_yaxis()


