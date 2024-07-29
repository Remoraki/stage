from grids import *
import numpy as np
import matplotlib.pyplot as plt
from findiffs import *
from utils import *
from scipy.linalg import expm


"""
Do one optimisation step
"""


def soft_heaviside(x, epsilon=0.1):
    return 0.5*(1 + (2/np.pi)*np.atan(x/epsilon))


def soft_delta(x, epsilon=0.1):
    return 0.5*(2/np.pi) * (1/(1 + (x/epsilon)**2)) / epsilon


def chi_funcs(A: GridForm2D, B: GridForm2D, heaviside_eps=0.5, sdf_iter=100, ):
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
    return chi_A, chi_B

def sym_diff(grid, chi_A, chi_B):
    return grid.integrate(np.square(chi_A-chi_B))


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


def plot_vector_field(X, Y, V):
    fig = plt.figure()
    fig.suptitle('Vector field')
    ax0 = plt.subplot(121)
    ax1 = plt.subplot(122)
    ax0.quiver(X, Y, V[:, :, 0], V[:, :, 1])
    ax1.imshow(np.linalg.norm(V, axis=-1))
    ax1.invert_yaxis()
    ax0.set_title('Vector field')
    ax1.set_title('Vector field norm')


def gram_matrix2d(shape: 'GridForm2D', chi):
    grid = shape.grid
    X = grid.X
    Y = grid.Y

    X2_plus_Y2 = X ** 2 + Y ** 2

    g = np.zeros((grid.shape[0], grid.shape[1], 4, 4))

    g[:, :, 0, 0] = chi * X2_plus_Y2
    g[:, :, 0, 1] = 0
    g[:, :, 0, 2] = chi * X
    g[:, :, 0, 3] = chi * Y

    g[:, :, 1, 0] = 0
    g[:, :, 1, 1] = chi * X2_plus_Y2
    g[:, :, 1, 2] = -chi * Y
    g[:, :, 1, 3] = chi * X

    g[:, :, 2, 0] = chi * X
    g[:, :, 2, 1] = -chi * Y
    g[:, :, 2, 2] = chi
    g[:, :, 2, 3] = 0

    g[:, :, 3, 0] = chi * Y
    g[:, :, 3, 1] = chi * X
    g[:, :, 3, 2] = 0
    g[:, :, 3, 3] = chi

    return grid.integrate(g)


def similarity_projection2d(A: GridForm2D, chi_A, v):
    Gram = gram_matrix2d(A, chi_A)
    Gram_inv = np.linalg.inv(Gram)
    grid = A.grid
    X = grid.X
    Y = grid.Y
    # Similarity space decomposition on the grid
    # B:(4,n,m,2) v:(n,m,2)
    B = np.array([np.transpose([X, Y]),
                  np.transpose([-Y, X]),
                  np.transpose([np.ones_like(X), np.zeros_like(X)]),
                  np.transpose([np.zeros_like(X), np.ones_like(X)])])
    # vectors scalar product (4,n,m)
    s = np.einsum('ijlm,kijml->kij', v[:, :, np.newaxis, :], B[:, :, :, :, np.newaxis])
    # vector fields scalar product (integral of vectors scalar product):(4)
    s = np.array([grid.integrate(s[i] * chi_A) for i in range(len(s))])
    # projection coordinates system solving
    lbda = np.matmul(Gram_inv, s)
    sigma = lbda[0] * np.eye(2)
    xi = lbda[1] * np.array([[0, -1], [1, 0]])
    u = np.array([lbda[2], lbda[3]])
    # Infinitesimal similarity
    dG = np.hstack((sigma + xi, u[:, np.newaxis]))
    dG = np.vstack((dG, [0, 0, 0]))
    return dG


def step(A: GridForm2D, B: GridForm2D, heaviside_eps=0.5, sdf_iter=100, descent_step=0.1):
    """
    Perform one optimisation step
    :param A: The model shape
    :param B: The target shape
    :param heaviside_eps: Epsilon value for heaviside
    :param sdf_iter: Number of iterations for sdf calculation
    :param descent_step: Value for gradient descent
    :return: The deformed shape
    """
    # Optimal vector field computation
    chi_A, chi_B = chi_funcs(A, B, heaviside_eps, sdf_iter)
    chi_diff = chi_A - chi_B
    phi_A = A.get_sdf(sdf_iter)
    grad_phi_A = sdf_gradient(A, sdf_iter)
    delta_phi_A = soft_delta(-phi_A, heaviside_eps)
    deform_field = -chi_diff[:, :, np.newaxis] * delta_phi_A[:, :, np.newaxis] * grad_phi_A
    # Projecting the vector field on the similarities space
    dG = similarity_projection2d(A, chi_A, deform_field)
    G = expm(descent_step * dG)
    # Computing the transformed shape
    GA = A.similarity(G)
    print('Sym diff : ' + str(sym_diff(A.grid, chi_A, chi_B)))
    print(G)
    return GA, G


def optimise(grid: Grid2D, A: GridForm2D, B: GridForm2D, nb_iter, heaviside_eps=0.5, sdf_iter=100, descent_step=0.2):
    plt.ion()
    fig = plt.figure()
    ax = plt.subplot()
    G = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    GA = A
    for i in range(nb_iter):
        GA = A.similarity(G)
        ax.clear()
        ax.contour(grid.X, grid.Y, GA.chi, levels=[0.5], colors='red')
        ax.contour(grid.X, grid.Y, B.chi, levels=[0.5], colors='blue')
        plt.draw()
        plt.pause(0.1)
        _, Gi = step(GA, B, heaviside_eps, sdf_iter, descent_step)
        G = np.matmul(Gi, G)

        plt.draw()


    plt.ioff()
    plt.show()

    return GA
















