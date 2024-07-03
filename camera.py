import numpy as np


class Camera:
    def __init__(self, f, rot, t, cu, cv):
        self.f = f
        self.rot = rot
        self.t = t
        self.cu = cu
        self.cv = cv
        self.K = [[f, 0, cu], [0, f, cv]]

    def projection(self, m):
        m_cam = (np.matmul(self.rot, m) + self.t)
        print(m_cam)
        p = np.matmul(self.K, m_cam / m_cam[2])
        return p
