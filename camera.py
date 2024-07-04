import numpy as np
import matplotlib.pyplot as plt


def get_camera_from_vector(pos, look_at, up, f=0.005, cu=0, cv=0):
    """
    Get a camera from its position and view angle in 3D world coordinates (y is depth and z is height)
    :param pos: the position of the camera
    :param look_at: the point the camera is looking at
    :param up: a vector that defines the 2D space that locks the y and z rotation
    :param f: the focal length of the camera
    :param cu: the u component of the optic center projection on the screen
    :param cv: the v component of the optic center projection on the screen
    :return:
    """
    forward = look_at - pos
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    rot = (np.vstack((right, forward, up)))
    t = - np.matmul(rot, pos)
    return Camera(rot, t, f, cu, cv)


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


class Camera:
    def __init__(self, rot, t, f=0.005, cu=0, cv=0, x_screen_size=0.01, y_screen_size=0.01, width=1000, height=1000,):
        """
        Creates a camera from all its parameters
        :param rot: The rotation matrix 3x3
        :param t: The translation vector 3x1
        :param f: the focal length of the camera
        :param cu: the u component of the optic center projection on the screen
        :param cv: the v component of the optic center projection on the screen
        :param x_screen_size: the x screen size of the camera
        :param y_screen_size: the y screen size of the camera
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
        self.resolution = [width,height]
        self.x_screen_size = x_screen_size
        self.y_screen_size = y_screen_size
        x_pixel_size = x_screen_size / width
        y_pixel_size = y_screen_size / height
        if x_pixel_size != y_pixel_size:
            print('Problem with screen proportions')
            raise Exception('Problem with screen proportions')
        self.pixel_size = x_pixel_size
        self.x_screen = np.linspace(-x_screen_size + self.pixel_size/2, x_screen_size - self.pixel_size/2, width)
        self.y_screen = np.linspace(-y_screen_size + self.pixel_size/2, y_screen_size - self.pixel_size/2, height)
        self.screen = []
        self.z_buffer = [[-1]*width]*height

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
            if not(abs(p[0]) > self.x_screen_size or abs(p[1]) > self.y_screen_size):
                return [p[0], p[1], z]
        return None

    def rasterize(self, p):
        """
        Get the indices of the pixel where the point is
        :param p: the 2D point in camera screen space
        :return: the indices in camera screen space
        """
        dx = np.abs(self.x_screen - p[0])
        dy = np.abs(self.y_screen - p[1])
        ix = np.argmin(dx)
        iy = np.argmin(dy)
        return [ix, iy, p[2]]

    def set_pixel(self, x, y, z, v):
        """
        Modifies a pixel value
        :param x: pixel x coordinate
        :param y: pixel y coordinate
        :param v: pixel value
        :return:
        """
        if z < self.z_buffer[x][y] or self.z_buffer[x][y] == -1:
            self.screen.append([x,y])
            self.z_buffer[x][y] = z
            return True
        return False

    def see_point(self, m):
        """
        Projects a point on camera screen
        :param m: 3D point in world space
        :return:
        """
        p = self.projection(m)
        if p is not None:
            r = self.rasterize(p)
            if self.set_pixel(r[0], r[1], r[2], 1):
                return True
        return False

    def get_plot(self):
        """
        Get a simple plot for camera screen rendering
        :return: fig, ax
        """
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')  # Aspect ratio 1:1
        ax.set_title('Camera screen')
        return fig, ax

    def render(self, ax):
        """
        Render the current screen on a given ax plot
        :param ax: the ax plot
        :return:
        """
        for [x, y] in self.screen:
            pixel = plt.Rectangle((x, y), 1, 1)
            ax.add_patch(pixel)

    def wipe(self):
        """
        Wipes the screen and the z_buffer
        :return:
        """
        self.screen = []
        self.z_buffer = [[-1] * self.width] * self.height







