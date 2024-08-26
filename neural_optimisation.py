import numpy as np
import torch as torch
import optimisation as opt
from neural_sdf.sdf2d import *
from utils import *
from findiffs import *
from scipy.linalg import expm


class NeuralOptimiser:
    def __init__(self, grid: Grid2D, device):
        self.grid = grid
        self.device = device

    def output(self, sdf: SDF2, G=np.eye(3)):
        points = self.grid.flatten(True)
        transformed = np.matmul(G, points)
        positions = np.transpose(transformed[0:2, :])
        encoded = encoding(torch.from_numpy(positions)).float().to(self.device)
        output = sdf(encoded).detach().cpu().numpy()
        return np.reshape(output, self.grid.shape)

    def chi_func(self, phi):
        chi = opt.soft_heaviside(-phi)
        return np.reshape(chi, self.grid.shape)

    def phi_grad(self, phi, iter=100):
        """
        Compute the gradient of the sdf of a form
        :param A: The form to compute from
        :param iter: The number of iterations for sdf calculation
        :return: A grid-shaped array of vectors
        """
        grad_phi_x = cdiff_x(phi)  # / grid.dS[0]
        grad_phi_y = cdiff_y(phi)  # / grid.dS[1]
        grad_phi = np.dstack((grad_phi_y, grad_phi_x))
        return grad_phi

    def similarity_projection2d(self, v, dmu):
        grid = self.grid
        X = grid.X
        Y = grid.Y
        # Similarity space decomposition on the grid
        # B:(4,n,m,2) v:(n,m,2)
        B = np.array([np.dstack((X, Y)),
                      np.dstack((-Y, X))])
        # vectors scalar product (4,n,m)
        s = np.einsum('ijlm,kijml->kij', v[:, :, np.newaxis, :], B[:, :, :, :, np.newaxis])
        sigma = grid.integrate(dmu * s[0]) * np.array([[1, 0], [0, 1]])
        xi = grid.integrate(dmu * s[1]) * np.array([[0, -1], [1, 0]])
        u = grid.integrate(dmu[:, :, np.newaxis] * v)

        dG = np.hstack((sigma + xi, u[:, np.newaxis]))
        dG = np.vstack((dG, [0, 0, 0]))
        return dG


    def optimal_vector_field(self, phi_A, phi_B):
        grad_phi_A = self.phi_grad(phi_A)
        grad_phi_B = self.phi_grad(phi_B)
        chi_A = self.chi_func(phi_A)
        chi_B = self.chi_func(phi_B)
        chi_diff = chi_A - chi_B
        delta_phi_A = opt.soft_delta(-phi_A)
        deform_field = -chi_diff[:, :, np.newaxis] * delta_phi_A[:, :, np.newaxis] * grad_phi_A
        return deform_field

    def step(self, A_sdf: SDF2, B_sdf: SDF2, Ginv, drawers, descent_step=0.1):
        phi_A = self.output(A_sdf, Ginv)
        phi_B = self.output(B_sdf)
        drawers[0].draw(phi_A.flatten())
        drawers[1].draw(phi_B.flatten())
        v = self.optimal_vector_field(phi_A, phi_B)
        drawers[2].draw(v)
        dG = self.similarity_projection2d(v, opt.soft_delta(-phi_A))
        G = expm(descent_step * dG)
        Ginv = np.linalg.inv(G)
        drawers[3].draw_on_chi(G, (phi_A <= 0).astype(float).flatten())
        return G, Ginv

    def optimise(self, A_sdf: SDF2, B_sdf: SDF2, nb_iter):
        Ginv = np.eye(3)
        G = np.eye(3)
        fig = plt.figure(figsize=(16, 8))
        A_drawer = SdfDrawer(self.grid, fig, 121, 'priori shape')
        B_drawer = SdfDrawer(self.grid, fig, 122, 'target shape')
        fig1 = plt.figure(figsize=(16, 8))
        grad_drawer = VectorFieldDrawer(self.grid, fig1, 131, 'transformation')
        g_drawer = SimDrawer(self.grid, fig1, 132, 'similarity')
        G_drawer = SimDrawer(self.grid, fig1, 133, 'total similarity')

        for i in range(nb_iter):
            Gi, Ginvi = self.step(A_sdf, B_sdf, Ginv, [A_drawer, B_drawer, grad_drawer, g_drawer])
            G = np.matmul(Gi, G)
            Ginv = np.matmul(Ginv, Ginvi)
        return G