import numpy as np
from matplotlib import pyplot as plt
from redistance import redistance
from scipy.interpolate import griddata
from typing import List


def merge_grids(grids: List['Grid2D'], square_output=True):
    """
    Rescale grids on a single same grid containing all of them
    :param square_output: Should the output grid be squared
    :param grids: A list of grids to merge: List[Grid2D]
    :return: A single grid containing all of them : Grid2D
    """
    bl = grids[0].bottom_left
    tr = grids[0].top_right
    min_x, max_x, min_y, max_y = bl[0], tr[0], bl[1], tr[1]
    res = grids[0].resolution
    for grid in grids:
        bl = grid.bottom_left
        tr = grid.top_right
        min_x, max_x, min_y, max_y = min(min_x, bl[0]), max(max_x, tr[0]), min(min_y, bl[1]), max(max_y, tr[1])
        res = min(res, grid.resolution)
    if square_output:
        dx = max_x - min_x
        dy = max_y - min_y
        if dy > dx:
            max_x += (dy - dx)/2
            min_x -= (dy - dx)/2
        elif dx > dy:
            max_y += (dx - dy)/2
            min_y -= (dx - dy)/2
    return Grid2D((min_x, min_y), (max_x, max_y), res)


def rescale_all(forms: List['GridForm2D'], border_width=0.2, square_output=True):
    """
    Rescale all forms on a single same grid perfectly scaled
    :param forms: A list of the forms to rescale: List[GridForm2D]
    :param square_output: Should the output grid be squared
    :param border_width: The width of the border around the forms
    :return: A tuple with the rescaled grid and a list of the rescaled forms: (Grid2D, List[GridForm2D])
    """
    grids = [form.scaled_grid(border_width, square_output) for form in forms]
    grid = merge_grids(grids, square_output)
    forms = [grid.form_interpolation(form.flatten()) for form in forms]
    return grid, forms


class Grid2D:
    def __init__(self, bottom_left, top_right, resolution):
        """
        A 2D grid to be shared across different calculations with different functions.
        :param bottom_left: the position of the bottom left corner: (2)
        :param top_right: the position of the top right corner: (2)
        :param resolution: The x and y resolution of the grid: (2)
        """
        width = top_right[0] - bottom_left[0]
        height = top_right[1] - bottom_left[1]
        if width <= 0 or height <= 0:
            raise ValueError('Grid2D: width and height must be strictly positive')
        self.dS = [width / resolution[0], height / resolution[1]]
        X = np.linspace(bottom_left[0] + self.dS[0] / 2, top_right[0] - self.dS[0] / 2, resolution[0])
        Y = np.linspace(bottom_left[1] + self.dS[1] / 2, top_right[1] + self.dS[0] / 2, resolution[1])
        self.bottom_left, self.top_right = bottom_left, top_right
        self.resolution = resolution
        self.X, self.Y = np.meshgrid(X, Y)
        self.shape = self.X.shape

    def form(self, values):
        """
        Get a form on the grid initialised with certain values
        :param values: A grid-sized array of values
        :return: A GridForm2D containing the values
        """
        return GridForm2D(self, values)

    def form_interpolation(self, points, method='linear'):
        """
        Create a grid form from anoter form
        :param method: The interpolation type (nearest, linear or cubic). Nearest can be used when interpolating on a
        lower resolution grid. Linear should be used when interpolating on a higher resolution grid.
        :param points: a (3,n) array of points (x,y,chi(x,y))
        :return: GridForm2D
        """
        if points.ndim != 2 or points.shape[0] != 3:
            raise ValueError('Grid2D: chiXY must be a (3,n) array of points')
        X, Y, chi = points[0, :], points[1, :], points[2, :]
        Z = griddata(np.transpose(np.vstack((X, Y))), chi, (self.X, self.Y), method, 0)
        return self.form(np.round(Z))

    def integrate(self, values):
        """
        Perform integration on the grid
        :param values: A grid-shaped array
        :return: The integral value
        """
        if (values.shape[0] != self.shape[0] or values.shape[1] != self.shape[1]
                or values.shape[0] != self.shape[0] or values.shape[1] != self.shape[1]):
            raise ValueError('Grid2D: grid and values shape must be the same')
        return np.sum(values, axis=(0, 1)) * self.dS[0] * self.dS[1]

    def integrate_on_form(self, form: 'GridForm2D', values):
        """
        Perform integration on a specific form
        :param form: the form on which to integrate: GridForm2D
        :param values: the values to integrate
        :return: The integral value
        """
        if (form.grid.shape[0] != self.shape[0] or form.grid.shape[1] != self.shape[1]
                or values.shape[0] != self.shape[0] or values.shape[1] != self.shape[1]):
            raise ValueError('Grid2D: grid shape must be the same')
        # broadcasting so we can have any value type we want on a grid
        dim = 3
        mask = form.chi
        while values.ndim >= dim:
            mask = mask[:, :, np.newaxis]
            dim += 1
        return np.sum(values * mask, axis=(0, 1)) * self.dS[0] * self.dS[1]

    def flatten(self, extended=False):
        X = self.X.flatten()
        Y = self.Y.flatten()
        if not extended:
            return np.vstack((X, Y))
        else:
            return np.vstack((X, Y, np.ones_like(X)))


class GridForm2D:
    def __init__(self, grid: 'Grid2D', chi):
        """
        A form defined on a specific 2D grid.
        :param grid: The grid to define the form on: Grid2D
        :param chi: The grid-shape characteristic function to define the form on, should be only 0 or 1.
        """
        if chi.shape[0] != grid.shape[0] or chi.shape[1] != grid.shape[1]:
            raise ValueError("grid and values must have the same shape")
        self.grid = grid
        self.chi = chi
        self.area = 0
        self.barycenter = None
        self.covariance_matrix = None
        self.sdf = None

    def plot_surface(self, ax, title='Characteristic function'):
        """
        Plot a surface representation of the form
        :param ax: The subplot on which to plot
        :param title: The title of the plot
        :return:
        """
        ax.plot_surface(self.grid.X, self.grid.Y, self.chi)
        ax.set_title(title)

    def plot_contour(self, ax: plt.Axes, title='Shape border', color='red'):
        """
        Plot a contour representation of the form
        :param ax: The subplot on which to plot
        :param title: The title of the plot
        :param color: The color of the contour
        :return:
        """
        ax.contour(self.grid.X, self.grid.Y, self.chi, levels=[0.5], colors=[color])
        ax.set_title(title)

    def plot_sdf_surface(self, ax: plt.Axes, iterations=100, title='sdf'):
        """
        Plot a surface representation of the form sdf
        :param ax: The subplot on which to plot
        :param iterations: The number of iterations for sdf calculations
        :param title: The title of the plot
        :return:
        """
        ax.plot_surface(self.grid.X, self.grid.Y, self.get_sdf(iterations))
        ax.set_title(title)

    def plot_sdf_im(self, ax: plt.Axes, iterations=100, title='sdf'):
        """
        Plot a 2D representation of the sdf
        :param ax: The subplot on which to plot
        :param iterations: The number of iterations for sdf calculations
        :param title: The title of the plot
        :return:
        """
        ax.imshow(self.get_sdf(iterations))
        ax.set_title(title)

    def integrate(self, values):
        """
        Integrate values on the form
        :param values: A grid-shaped array of values
        :return: The value of the integral on the form
        """
        return self.grid.integrate_on_form(self, values)

    def get_area(self):
        """
        Get the area of the form
        :return: Area
        """
        if self.area == 0:
            self.area = self.integrate(self.chi)
        return self.area

    def get_barycenter(self):
        """
        Get the barycenter of the form
        :return: Barycenter: (2)
        """
        if self.barycenter is None:
            values = np.stack((self.grid.X, self.grid.Y), axis=-1)
            self.barycenter = self.integrate(values) / self.get_area()
        return self.barycenter

    def get_covariance_matrix(self):
        """
        Get the covariance matrix of the form
        :return: Covariance matrix: (2,2)
        """
        if self.covariance_matrix is None:
            centered_points = np.stack((self.grid.X, self.grid.Y), axis=-1) - self.get_barycenter()
            values = np.einsum('ijk,ijl->ijkl', centered_points, centered_points)
            self.covariance_matrix = self.integrate(values) / self.get_area()
        return self.covariance_matrix

    def get_coords(self, extended=False):
        """
        Get a list of the points of the form
        :param extended: Should the coordinates to be extended with a 1
        :return: A (2/3,n) array containing the coordinates
        """
        points = np.array(np.where(self.chi))
        X = self.grid.X[points[0, :], points[1, :]]
        Y = self.grid.Y[points[0, :], points[1, :]]
        if extended:
            return np.vstack((X, Y, np.ones_like(X)))
        return np.vstack((X, Y))

    def flatten(self):
        """
        Get the flattened values defining the form
        :return: A 3-tuple of (1,n) arrays (X,Y,Chi)
        """
        return np.vstack((self.grid.X.flatten()[np.newaxis, :],
                          self.grid.Y.flatten()[np.newaxis, :],
                          self.chi.flatten()[np.newaxis, :]))

    def similarity(self, G):
        """
        Apply a similarity transformation to the shape
        :param G: A similarity matrix (R,T): (2,3)
        :return: The transformed shape: GridForm2D
        """
        X, Y, chi = self.flatten()
        XY = np.vstack((X[np.newaxis, :], Y[np.newaxis, :], np.ones((1, len(X)))))
        GXY = np.matmul(G, XY)
        GX = GXY[0, :]
        GY = GXY[1, :]
        return self.grid.form_interpolation(np.array([GX, GY, chi]))

    def deform(self, vector_field, dt=0.01):
        """
        Deform the form based on a vector field
        :param vector_field: A grid-shaped array of 2D vectors: (n,m,2)
        :param dt: The length of the time step
        :return: The deformed shape: GridForm2D
        """
        X = (self.grid.X + dt * vector_field[:, :, 0]).flatten()
        Y = (self.grid.Y + dt * vector_field[:, :, 1]).flatten()
        chi = self.chi.flatten()
        return self.grid.form_interpolation(np.vstack((X[np.newaxis, :], Y[np.newaxis, :], chi[np.newaxis, :])))

    def get_sdf(self, iterations=100):
        """
        Get tbe sdf represented by the form
        :param iterations: The number of iterations with which to calculate the sdf
        :return: A grid-shaped sdf array: (n,m)
        """
        if self.sdf is None:
            self.sdf = redistance(-self.chi + 0.5, iterations)
        return self.sdf

    def scaled_grid(self, border_width=0.1, square_output=True):
        """
        Get a grid perfectly scaled on the form with the same resolution
        :param border_width: The forced width of the border of the grid, so that the shape won't get to close to it
        :param square_output: Should the scaled grid be squared
        :return:
        """
        points = self.get_coords()
        if points.shape[1] == 0:
            return self.grid
        min_x = min(points[0, :]) - border_width*self.grid.dS[0]*self.grid.resolution[0]
        max_x = max(points[0, :]) + border_width*self.grid.dS[0]*self.grid.resolution[0]
        min_y = min(points[1, :]) - border_width*self.grid.dS[1]*self.grid.resolution[1]
        max_y = max(points[1, :]) + border_width*self.grid.dS[1]*self.grid.resolution[1]
        if square_output:
            dx = max_x - min_x
            dy = max_y - min_y
            if dy > dx:
                max_x += (dy - dx) / 2
                min_x -= (dy - dx) / 2
            elif dx > dy:
                max_y += (dx - dy) / 2
                min_y -= (dx - dy) / 2
        grid = Grid2D((min_x, min_y), (max_x, max_y), self.grid.resolution)
        return grid

    def rescale(self, border_width=0.1, square_output=True):
        """
        Rescale the form on a new grid
        :param border_width: The forced width of the border of the grid, so that the shape won't get to close to it
        :param square_output: Should the new grid be squared
        :return: A tuple with the form and the grid: (GridForm2D, Grid2D)
        """
        grid = self.scaled_grid(border_width, square_output)
        nf = grid.form_interpolation(self.flatten())
        return nf, grid




