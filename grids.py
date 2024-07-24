import numpy as np
from matplotlib import pyplot as plt
from redistance import redistance
from scipy.interpolate import griddata


def rescale_grids(grids):
    bl = grids[0].bottom_left
    tr = grids[0].top_right
    min_x, max_x, min_y, max_y = bl[0], tr[0], bl[1], tr[1]
    res = grids[0].resolution
    for grid in grids:
        bl = grid.bottom_left
        tr = grid.top_right
        min_x, max_x, min_y, max_y = min(min_x, bl[0]), max(max_x, tr[0]), min(min_y, bl[1]), max(max_y, tr[1])
        res = min(res, grid.resolution)
    return Grid2D((min_x, min_y), (max_x, max_y), res)


def rescale_all(forms):
    grids = []
    for form in forms:
        grids.append(form.scaled_grid())
    grid = rescale_grids(grids)
    for i in range(len(grids)):
        forms[i] = grid.form_interpolation(forms[i].get_coords())
    return grid, forms


class Grid2D:
    def __init__(self, bottom_left, top_right, resolution):
        """
        A 2D grid to be shared across different calculations with different functions.
        :param bottom_left: the position of the bottom left corner
        :param top_right: the position of the top right corner
        :param resolution: The x and y resolution of the grid
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

    def null_form(self):
        return GridForm2D(self, np.zeros(self.shape))

    def form(self, values):
        return GridForm2D(self, values)

    def form_interpolation(self, points, method='linear'):
        """
        Create a grid form from a set of points
        :param method: The interpolation type (nearest, linear or cubic). Nearest can be used when interpolating on a
        lower resolution grid. Linear should be used when interpolating on a higher resolution grid.
        :param points: a (n,2) array of points
        :return: A GridForm2D
        """
        if points.ndim != 2 or points.shape[0] != 2:
            raise ValueError('Grid2D: points must be a (2,n) array of points')
        Z = griddata(np.transpose(points), np.ones(points.shape[1]), (self.X, self.Y), method, 0)
        return self.form(Z)

    def integrate(self, form, values):
        """
        Perform integration on the grid
        :param form: the form on which to integrate
        :param values: the values of the grid
        :return: the value of the integral
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

    def rotate(self, form, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        coords = form.get_coords()
        rotated_coords = np.matmul(R, coords)
        return self.form_interpolation(rotated_coords)

    def translate(self, form, t):
        coords = form.get_coords()
        translated_coords = coords + t
        return self.form_interpolation(translated_coords)


class GridForm2D:
    def __init__(self, grid, chi):
        """
        A form defined on a specific 2D grid.
        :param grid: The grid to define the form on
        :param chi: The characteristic function to define the form on, should be only 0 or 1.
        """
        if chi.shape[0] != grid.shape[0] or chi.shape[1] != grid.shape[1]:
            raise ValueError("grid and values must have the same shape")
        self.grid = grid
        self.chi = chi
        self.area = 0
        self.barycenter = None
        self.covariance_matrix = None
        self.sdf = None

    def plot_surface(self, ax):
        ax.plot_surface(self.grid.X, self.grid.Y, self.chi)
        ax.set_title('Characteristic function')

    def plot_sdf(self, ax, iterations=100):
        ax.plot_surface(self.grid.X, self.grid.Y, self.get_sdf(iterations))
        ax.set_title('SDF')

    def plot_scatter(self, ax: plt.subplot, s=1, c='b'):
        mask = np.where(self.chi)
        ax.scatter(self.grid.X[mask], self.grid.Y[mask], s, c)
        ax.set_xlim([self.grid.X[0, 0], self.grid.X[-1, -1]])
        ax.set_ylim([self.grid.Y[0, 0], self.grid.Y[-1, -1]])

    def __reset(self):
        self.area = 0
        self.barycenter = None
        self.covariance_matrix = None
        self.sdf = None

    def integrate(self, values):
        return self.grid.integrate(self, values)

    def get_area(self):
        if self.area == 0:
            self.area = self.integrate(self.chi)
        return self.area

    def get_barycenter(self):
        if self.barycenter is None:
            values = np.stack((self.grid.X, self.grid.Y), axis=-1)
            self.barycenter = self.integrate(values) / self.get_area()
        return self.barycenter

    def get_covariance_matrix(self):
        if self.covariance_matrix is None:
            centered_points = np.stack((self.grid.X, self.grid.Y), axis=-1) - self.get_barycenter()
            values = np.einsum('ijk,ijl->ijkl', centered_points, centered_points)
            self.covariance_matrix = self.integrate(values) / self.get_area()
        return self.covariance_matrix

    def get_coords(self, extended=False):
        """
        Get coordinates of the points of the form
        :param extended: Should the coordinates to be extended with a 1
        :return: A (2,n) array containing the coordinates
        """
        points = np.array(np.where(self.chi))
        X = self.grid.X[points[0, :], points[1, :]]
        Y = self.grid.Y[points[0, :], points[1, :]]
        if extended:
            return np.vstack((X, Y, np.ones_like(X)))
        return np.vstack((X, Y))

    def get_sdf(self, iterations=100):
        if self.sdf is None:
            self.sdf = redistance(-self.chi + 0.5, iterations)
        return self.sdf

    def scaled_grid(self):
        points = self.get_coords()
        min_x = min(points[0, :]) - self.grid.dS[0]
        max_x = max(points[0, :]) + self.grid.dS[0]
        min_y = min(points[1, :]) - self.grid.dS[1]
        max_y = max(points[1, :]) + self.grid.dS[1]
        grid = Grid2D((min_x, min_y), (max_x, max_y), self.grid.resolution)
        return grid



