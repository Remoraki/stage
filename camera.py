import numpy as np
import matplotlib.pyplot as plt


def get_camera_from_vector(name, pos, look_at, up, f=0.005, cu=0, cv=0, screen_scaling=10000, width=120, height=100,):
    """
    Get a camera from its position and view angle in 3D world coordinates (y is depth and z is height)
    :param name: the name of the camera
    :param pos: the position of the camera
    :param look_at: the point the camera is looking at
    :param up: a vector that defines the 2D space that locks the y and z rotation
    :param f: the focal length of the camera
    :param cu: the u component of the optic center projection on the screen
    :param cv: the v component of the optic center projection on the screen
    :return:
    """
    pos = np.array(pos)
    look_at = np.array(look_at)
    up = np.array(up)
    forward = look_at - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    rot = (np.vstack((right, forward, up)))
    t = - np.matmul(rot, pos)
    return Camera(name, rot, t, f, cu, cv, screen_scaling, width, height)

# ---fonctions à utiliser plus tard si besoin de changer de repère---


def to_world_space(m):
    """
    Convert 3D standard space(x:right,y:down,z:forward) to world space(x:right,y:forward,z:up).
    :param m: 3D point
    :return: 3D point
    """
    return [m[0], m[2], -m[1]]


def to_screen_space(p):
    """
    Convert 2D coordinates in camera screen space(u:right,v:up) to 2D standard screen space(u:down,v:right)
    :param p:
    :return:
    """
    return [-p[1], p[0]]
# ------


class Camera:
    """
    This class can be used to represent a 3D camera.
    You can project 3D points on its screen, and render it.
    For now, only point rasterization is supported, will need to add surface rasterization,
    like mesh triangles later.
    """
    def __init__(self, name, rot, t,
                 f=0.01, cu=0, cv=0,
                 screen_scaling=10000, width=120, height=100):
        """
        Creates a camera from all its parameters
        :param rot: The rotation matrix 3x3
        :param t: The translation vector 3x1
        :param f: the focal length of the camera
        :param cu: the u component of the optic center projection on the screen
        :param cv: the v component of the optic center projection on the screen
        :param width: the width of the camera screen
        :param height: the height of the camera screen
        """
        self.rot = rot
        self.t = t
        self.f = f
        self.cu = cu
        self.cv = cv
        self.K = [[f, 0, cu], [0, f, cv]]
        self.width = width
        self.height = height
        self.x_screen_size = width / screen_scaling
        self.y_screen_size = height / screen_scaling
        self.pixel_size = 1 / screen_scaling
        self.x_screen = np.linspace(
            -self.x_screen_size + self.pixel_size/2, self.x_screen_size - self.pixel_size/2, width)
        self.y_screen = np.linspace(
            -self.y_screen_size + self.pixel_size/2, self.y_screen_size - self.pixel_size/2, height)
        self.x_screen_grid, self.y_screen_grid = np.meshgrid(self.x_screen, self.y_screen)
        self.pixel_values = np.array([[0.]*width]*height)
        self.screen = set()
        self.name = name

    def projection(self, m):
        """
        Project a point on a camera screen
        :param m: 3D on world space point for which y is depth and z is height
        :return: A 2D point on the camera screen space for which u is right and v is up, and the depth value
        """
        # putting the point in camera axes
        m_cam = (np.matmul(self.rot, m) + self.t)
        z = m_cam[1]
        # switching to image axes where z is the depth
        m_im = [m_cam[0], m_cam[2], m_cam[1]]
        # retrieving screen coordinates
        if z > 0:
            p = np.matmul(self.K, m_im) / z
            if not (abs(p[0]) > self.x_screen_size or abs(p[1]) > self.y_screen_size):
                return p[0], p[1], z
        return None

    def rasterize(self, p):
        """
        Get the indices of the pixel where the point is
        :param p: the 2D point in camera screen space with its depth value
        :return: the indices in camera screen space, and the depth value
        """
        dx = np.abs(self.x_screen - p[0])
        dy = np.abs(self.y_screen - p[1])
        ix = np.argmin(dx)
        iy = np.argmin(dy)
        return iy, ix

    def gaussian_splatting(self, p):
        """
        Apply gaussian_splatting rasterisation of a given point on screen
        :param p: the 2D point in camera screen space with its depth value
        :return: the indices in camera screen space, and the depth value
        """
        dx = np.abs(self.x_screen_grid - p[0])
        dy = np.abs(self.y_screen_grid - p[1])
        d = np.exp((-np.square(dx) - np.square(dy)) * (5000*p[2])**2 / 2)
        self.pixel_values += d

    def get_pixel_value(self, p):
        """
        Tells if a pixel is lit or not
        :param p: the pixel coordinates (x,y)
        :return: the current value of the pixel
        """
        return self.pixel_values[p]

    def place_point(self, m):
        """
        Place a point on camera screen
        :param m: 3D point in world space
        """
        p = self.projection(m)
        if p is not None:
            self.gaussian_splatting(p)

    def get_plot(self):
        """
        Get a simple plot for camera screen rendering
        :return: fig, ax
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')  # Aspect ratio 1:1
        ax.set_title('Screen from : ' + self.name)
        return fig, ax

    def render(self):
        """
        Render the current screen
        :return:
        """
        _, ax = self.get_plot()
        grid_i, grid_j = np.meshgrid(range(self.width), range(self.height))
        for i in range(self.height):
            for j in range(self.width):
                if self.pixel_values[i, j] >= 0.1:
                    pixel = plt.Rectangle((j, i), 1, 1)
                    ax.add_patch(pixel)

    def wipe(self):
        """
        Wipes the screen and the z_buffer
        :return:
        """
        self.screen = set()








