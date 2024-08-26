import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def rasterize(p):
    """
    Get the indices of the pixel where the point is
    :param p: the 2D point in camera screen space
    :return: the indices in camera screen space
    """
    return round(p[0]), round(p[1])


class Camera:
    """
    This class can be used to represent a 3D camera.
    You can project 3D points on its screen, and render it, or feed it a preexisting image in pixel_values.
    """
    def __init__(self, name, rot, t, K, width=120, height=100):
        """
        Creates a camera from all its parameters
        :param rot: The rotation matrix 3x3
        :param t: The translation vector 3x1
        :param K: the calibration matrix
        :param width: the width of the camera screen
        :param height: the height of the camera screen
        """
        self.rot = rot
        self.t = t
        self.K = K
        self.width = width
        self.height = height
        self.pixel_values = np.array([[0.]*width]*height)
        self.screen = set()
        self.name = name

    def projection(self, m):
        """
        Project a point on the camera screen
        :param m: 3D point in world space
        :return: A 2D point on the camera screen space
        """
        # putting the point in camera space
        m_cam = np.matmul(self.rot, m) + self.t
        z = m_cam[2]
        # retrieving screen coordinates
        if z > 0:
            p = np.matmul(self.K, m_cam) / z
            if 0 <= p[0] < self.height - 1 and 0 <= p[1] < self.width - 1:
                return p[0], p[1], z
        return None

    def projection_matrix(self, M, only_inside=True):
        """
        Project a set of 3D points on the camera screen
        :param only_inside: If the function should return all projections or only the one inside the screen
        :param M: an 3*n matrix that contains all the points
        :return: a 2*n matrix that contains all the pixels corresponding to the projection,
        and the mask of all the pixels that are inside the screen
        """
        t = self.t[:, np.newaxis]
        M_cam = np.matmul(self.K, np.matmul(self.rot, M) + t)
        Z = np.copy(M_cam[2, :])
        M_cam /= Z
        P = np.round(M_cam).astype(int)
        lower_bound = np.array([[0], [0], [0]])
        upper_bound = np.array([[self.height-1], [self.width-1], [100000]])
        mask = np.logical_and(lower_bound <= M_cam, M_cam < upper_bound)
        mask = np.all(mask, axis=0)
        if only_inside:
            return P[:, mask], np.where(mask)[0]
        else:
            return P, np.where(mask)[0]

    def set_pixel(self, p, v):
        """
        Modifies a pixel value
        :param p: the pixel coordinates (x,y)
        :param v: pixel value
        :return:
        """
        self.pixel_values[p[1], p[0]] = v

    def get_pixel_value(self, p):
        """
        Tells if a pixel is lit or not
        :param p: the pixel coordinates
        :return:
        """
        if np.sum(self.pixel_values[p[1], p[0]]) > 0:
            return True
        else:
            return False

    def get_bounding_box(self, X, Y):
        x_min, x_max = np.min(X), np.max(X)
        y_min, y_max = np.min(Y), np.max(Y)
        x_min = min(max(x_min, 0), self.height - 1)
        x_max = min(max(x_max, 0), self.height - 1)
        y_min = min(max(y_min, 0), self.width - 1)
        y_max = min(max(y_max, 0), self.width - 1)
        top_left = (x_min, y_min)
        bottom_right = (x_max, y_max)
        return top_left, bottom_right

    def scan_points(self, X, Y):
        """
        Looks if the bounding box of the points contains any lit pixel
        :param X:
        :param Y:
        :return: -1 if the box is fully outside, 1 if it is fully inside, and 0 otherwise
        """
        top_left, top_right = self.get_bounding_box(X, Y)
        return self.scan_box(top_left, top_right)

    def scan_box(self, top_left, bottom_right):
        """
        Looks if the box contains any lit pixel
        :param top_left: The top left corner of the box in camera screen space
        :param bottom_right: The top right corner of the box in camera screen space
        :return: -1 if the box is fully outside, 1 if it is fully inside, and 0 otherwise
        """
        values = self.pixel_values[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0], :]
        pixel_values = np.sum(values, 2)
        nb_of_lit_pixels = np.sum(pixel_values > 0)
        if nb_of_lit_pixels == 0:
            return -1
        if nb_of_lit_pixels == pixel_values.shape[0] * pixel_values.shape[1]:
            return 1
        return 0

    def wipe(self):
        """
        Wipes the screen
        :return:
        """
        self.pixel_values = np.array([[]*self.width]*self.height)








