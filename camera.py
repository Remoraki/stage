import numpy as np


def get_camera_from_vector(pos, look_at, up, f, cu, cv):
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


class Camera:
    def __init__(self, rot, t, f, cu, cv):
        """
        Creates a camera from all its parameters
        :param rot: The rotation matrix
        :param t: The translation vector
        :param f: the focal length of the camera
        :param cu: the u component of the optic center projection on the screen
        :param cv: the v component of the optic center projection on the screen
        """
        self.rot = rot
        self.t = t
        self.f = f
        self.cu = cu
        self.cv = cv
        self.K = [[f, 0, cu], [0, f, cv]]

    def projection(self, m):
        """
        Project a point on a camera screen
        :param m: 3D point for which y is depth and z is height
        :return: A 2D point on the camera screen for which u is right and v is up
        """
        # putting the point in camera axes
        m_cam = (np.matmul(self.rot, m) + self.t)
        z = m_cam[1]
        # switching to image axes where z is the depth
        m_im = [m_cam[0], m_cam[2], m_cam[1]]
        # retrieving screen coordinates
        if z > 0:
            p = np.matmul(self.K, m_im) / z
        else:
            p = None
        print('proj')
        print(m)
        print(m_cam)
        print(m_im)
        print(p)
        return p
