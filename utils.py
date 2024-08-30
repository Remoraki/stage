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
    np.save('Data/A0', A0.chi)
    np.save('Data/grid0', grid_dic, True)


def load_init():
    grid_info = np.load('Data/grid0.npy', allow_pickle=True).item()
    grid = Grid2D(grid_info['bl'], grid_info['tr'], grid_info['res'])
    A0 = grid.form(np.load('Data/A0.npy'))
    return A0


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
        if A.grid != B.grid:
            grid, [A, B] = rescale_all([A, B], border_width, square_output)
        else:
            grid = B.grid
        np.save('Data/B', B.chi)
        grid_dic = {'res': grid.resolution, 'bl': grid.bottom_left, 'tr': grid.top_right}
        np.save('Data/grid', grid_dic, True)
    np.save('Data/A', A.chi)


def load_current(load_all=False):
    """
    Load the current state, with the model shape, the target shape, and their common grid if specified
    :param load_all: Do you need to all target shape and common grid as well
    :return: If load_all: (grid, A, B), else:(A)
    """
    grid_info = np.load('Data/grid.npy', allow_pickle=True).item()
    grid = Grid2D(grid_info['bl'], grid_info['tr'], grid_info['res'])
    A = grid.form(np.load('Data/A.npy'))
    if load_all:
        B = grid.form(np.load('Data/B.npy'))
        return grid, A, B
    else:
        return A


class Drawer:
    def __init__(self, grid, fig, index, title=""):
        plt.ion()
        self.fig = fig
        self.ax = plt.subplot(index)
        self.grid = grid
        self.title = title
        self.ax.set_title(self.title)


class NullDrawer:
    def draw(self, A):
        return

    def draw_on_chi(self, G, chi):
        return


class ContourDrawer(Drawer):
    def draw(self, A, B=None):
        grid = self.grid
        ax = self.ax
        ax.clear()
        ax.contour(grid.X, grid.Y, A.chi, levels=[0.5], colors='red')
        if B is not None:
            ax.contour(grid.X, grid.Y, B.chi, levels=[0.5], colors='blue')
        self.ax.set_title(self.title)
        plt.draw()
        plt.pause(0.01)


class VectorFieldDrawer(Drawer):
    def __init__(self, grid, fig, index, title="", scale=1.0):
        super().__init__(grid, fig, index)
        self.scale = scale

    def draw(self, v):
        grid = self.grid
        ax = self.ax
        ax.clear()
        ax.quiver(grid.X, grid.Y, v[:, :, 0], v[:, :, 1], scale=self.scale)
        self.ax.set_title(self.title)
        plt.draw()
        plt.pause(0.01)


class ScalarFieldDrawer(Drawer):
    def draw(self, s):
        grid = self.grid
        ax = self.ax
        ax.clear()
        ax.imshow(s)
        ax.invert_yaxis()
        self.ax.set_title(self.title)
        plt.draw()
        plt.pause(0.01)


class SimDrawer(VectorFieldDrawer):
    def __init__(self, grid, fig, index, title="", scale=1.0):
        super().__init__(grid, fig, index, title, scale)
        self.sim_grid = None

    def draw_on_shape(self, G, A: GridForm2D):
        self.draw_on_chi(A.chi.flatten())

    def draw_on_chi(self, G, chi):
        X = self.grid.X.flatten()
        Y = self.grid.Y.flatten()
        Z = np.ones_like(X)
        P = np.vstack((X, Y, Z))
        P = np.matmul(G, P)
        NX = P[0, :]
        NY = P[1, :]
        DX = (NX - X) * chi
        DY = (NY - Y) * chi
        gDX = np.reshape(DX, self.grid.X.shape)
        gDY = np.reshape(DY, self.grid.Y.shape)
        self.sim_grid = np.dstack((gDX, gDY))
        super().draw(self.sim_grid)




def draw_form_contour(form: GridForm2D, title='form', color='red', scaled=False, squared_scale=True, fig=None):
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


