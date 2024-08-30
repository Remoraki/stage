import torch
import numpy as np
from numpy.linalg import eig

from neural_sdf.sdf2d import encoding
from neural_sdf.net import SDFNet
from grids import Grid2D
import initialisation as init

class NeuralInitializer:
    """
    Class used to initialise an optimisation alignment problem
    """
    def __init__(self, grid: Grid2D, device, A_sdf: SDFNet, B_sdf: SDFNet):
        self.grid = grid
        self.device = device
        self.G = np.eye(3)
        self.A_sdf = A_sdf
        self.B_sdf = B_sdf
        self.G0 = np.eye(3)


    def A_output(self, G=np.eye(3)):
        """
        Get the output of the model sdf transformed by a similarity
        :param G: The similarity matrix to apply to the sdf
        :return: A tensor of the output
        """
        return self.__output(self.A_sdf, np.matmul(G, self.G0))

    def B_output(self):
        """
        Get the output of the target sdf
        :return: A tensor of the output
        """
        return self.__output(self.B_sdf)


    def __output(self, sdf: SDFNet, G=np.eye(3)):
        """
        Get the output of a sdf transformed by a similarity
        :param sdf: The sdf to get the output from
        :param G: The similarity matrix to apply to the sdf
        :return: The output
        """
        sigma = G[0, 0]
        Ginv = np.linalg.inv(G)
        points = self.grid.flatten(True)
        transformed = np.matmul(Ginv, points)
        positions = np.transpose(transformed[0:2, :])
        encoded = encoding(torch.from_numpy(positions)).float().to(self.device)
        output = 1 / sigma * sdf(encoded).detach().cpu().numpy()
        return np.reshape(output, self.grid.shape)

    def moments(self, sdf: SDFNet):
        """
        Compute the moments of an sdf shape
        :param sdf: The sdf to use
        :return: (area, barycenter, covariance)
        """
        output = self.__output(sdf)
        form = self.grid.form(output <= 0)
        return init.moments(form)

    def align(self):
        """
        Perform the alignment of the model and target shape based on the moments
        :return: the alignment similarity æatrix
        """
        area_A, barycenter_A, cov_A = self.moments(self.A_sdf)
        area_B, barycenter_B, cov_B = self.moments(self.B_sdf)
        eig_values_B, eig_vectors_B = eig(cov_B)
        eig_values_A, eig_vectors_A = eig(cov_A)
        ei_values_A, eig_values_B = np.real(eig_values_A), np.real(eig_values_B)
        t = barycenter_B - barycenter_A
        sigma = np.sqrt(np.sum(eig_values_B)) / np.sqrt(np.sum(eig_values_A))
        R = eig_vectors_B
        GAB = np.hstack((sigma * R, t[:, np.newaxis]))
        GAB = np.vstack((GAB, np.array([0, 0, 1])))
        self.G0 = GAB
        return GAB  # It's a me, GAB

    def apply_similarity(self, G):
        """
        Apply an transformation on the model sdf
        :param G: The similarity matrix
        :return:
        """
        self.G0 = np.matmul(G, self.G)
