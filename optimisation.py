from grids import *
import numpy as np
import matplotlib.pyplot as plt
from findiffs import *
from utils import *


"""
Do one optimisation step
"""


def soft_heaviside(x, epsilon=0.1):
    return 0.5*(1 + (2/np.pi)*np.atan(x/epsilon))


def soft_delta(x, epsilon=0.1):
    return 0.5*(2/np.pi) * (1/(1 + (x/epsilon)**2)) / epsilon


def sym_diff(A: GridForm2D, B: GridForm2D):
    """
    Compute the symmetric difference between two forms
    :param A: GridForm2D
    :param B: GridForm2D
    :return: A grid-shaped symmetric difference
    """
    phi_A = A.get_sdf(sdf_iter)
    phi_B = B.get_sdf(sdf_iter)
    chi_A = soft_heaviside(-phi_A, heaviside_eps)
    chi_B = soft_heaviside(-phi_B, heaviside_eps)
    sym_diff = chi_A - chi_B
    return sym_diff


def sdf_gradient(A: GridForm2D, iter=100):
    """
    Compute the gradient of the sdf of a form
    :param A: The form to compute from
    :param iter: The number of iterations for sdf calculation
    :return: A grid-shaped array of vectors
    """
    phi_A = A.get_sdf(iter)
    grad_phi_A_x = cdiff_x(phi_A)  # / grid.dS[0]
    grad_phi_A_y = cdiff_y(phi_A)  # / grid.dS[1]
    grad_phi_A = np.stack((grad_phi_A_x, grad_phi_A_y), axis=-1)
    return grad_phi_A

def plot_vector_field(V):
    fig = plt.figure()
    fig.suptitle('Vector field')
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)
    ax0.quiver(A.grid.X, A.grid.Y, V[:, :, 0], V[:, :, 1])
    ax1.imshow(np.linalg.norm(V, axis=-1))
    ax1.invert_yaxis()
    ax0.set_title('Vector field')
    ax1.set_title('Vector field norm')


def step(A, B, heaviside_eps=0.5, sdf_iter=100, descent_step=0.1):
    """
    Perform one optimisation step
    :param A: The model shape
    :param B: The target shape
    :param heaviside_eps: Epsilon value for heaviside
    :param sdf_iter: Number of iterations for sdf calculation
    :param descent_step: Value for gradient descent
    :param plot_shape: Plot the shape
    :param plot_vector_field: Plot the vector field
    :return: The deformed shape
    """
    sd = sym_diff(A, B)
    phi_A = A.get_sdf(sdf_iter)
    grad_phi_A = sdf_gradient(A, sdf_iter)
    delta_phi_A = soft_delta(-phi_A, heaviside_eps)
    deform_field = -sd[:, :, np.newaxis] * delta_phi_A[:, :, np.newaxis] * grad_phi_A
    GA = A.deform(deform_field, descent_step)
    return GA, deform_field


if __name__ == '__main__':
    # Constants
    nb_iter = 10
    heaviside_eps = 0.5
    sdf_iter = 500
    descent_step = 0.2
    # Init
    grid, A, B = load_current(True)
    # Iter
    GA = A
    for i in range(nb_iter):
        GA, V = step(GA, B, heaviside_eps, sdf_iter, descent_step)
        sd = sym_diff(GA, B)
        print('Sym diff : ' + str(grid.integrate(np.square(sd))))
        #plot_vector_field(V)
        plot_form_contour(GA)
        plt.show()
    plot_form_contour(A, 'Original shape')
    plot_form_contour(GA, 'Transformed shape')
    plot_form_contour(B, 'Target shape')
    plt.show()














