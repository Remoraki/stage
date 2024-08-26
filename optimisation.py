from grids import *
import numpy as np
import matplotlib.pyplot as plt
from findiffs import *
from utils import *
from scipy.linalg import expm



def soft_heaviside(x, epsilon=0.5):
    return 0.5*(1 + (2/np.pi)*np.arctan(x/epsilon))


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
    grad_phi_A = np.dstack((grad_phi_A_y, grad_phi_A_x))
    return grad_phi_A


def optimal_vector_field(A: GridForm2D, B: GridForm2D, heaviside_eps=0.5, sdf_iter=100, drawers=None):
    chi_A, chi_B = chi_funcs(A, B, heaviside_eps, sdf_iter)
    chi_diff = chi_A - chi_B
    phi_A = A.get_sdf(sdf_iter)
    grad_phi_A = sdf_gradient(A, sdf_iter)
    delta_phi_A = soft_delta(-phi_A, heaviside_eps)
    deform_field = -chi_diff[:, :, np.newaxis] * delta_phi_A[:, :, np.newaxis] * grad_phi_A
    if drawers is not None:
        drawers[0].draw(deform_field)
    return deform_field


def gram_matrix2d(shape: 'GridForm2D', heaviside_eps):
    """
    Get the gram matrix corresponding to a 2D shape similarity space
    :param shape: The shape to compute the gram matrix from: GridForm2D
    :return: The 2d gram matrix (4,4)
    """
    grid = shape.grid
    X = grid.X
    Y = grid.Y
    m = soft_delta(shape.get_sdf(), heaviside_eps)
    X2_plus_Y2 = X ** 2 + Y ** 2

    g = np.zeros((grid.shape[0], grid.shape[1], 4, 4))

    g[:, :, 0, 0] = m * X2_plus_Y2
    g[:, :, 0, 1] = 0
    g[:, :, 0, 2] = m * X
    g[:, :, 0, 3] = m * Y

    g[:, :, 1, 0] = 0
    g[:, :, 1, 1] = m * X2_plus_Y2
    g[:, :, 1, 2] = -m * Y
    g[:, :, 1, 3] = m * X

    g[:, :, 2, 0] = m * X
    g[:, :, 2, 1] = -m * Y
    g[:, :, 2, 2] = m
    g[:, :, 2, 3] = 0

    g[:, :, 3, 0] = m * Y
    g[:, :, 3, 1] = m * X
    g[:, :, 3, 2] = 0
    g[:, :, 3, 3] = m

    return grid.integrate(g)


def similarity_projection2d_old(A: GridForm2D, v, heaviside_eps):
    """
    Project a vector field on a 2d shape similarity space
    :param A: The shape
    :param v: The vector field
    :param heaviside_eps: The epsilon for heaviside calculations
    :return: The infinitesimal similarity, result of the projection
    """
    Gram = gram_matrix2d(A, heaviside_eps)
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
    s = np.array([grid.integrate(s[i] * soft_delta(A.get_sdf(), heaviside_eps)) for i in range(len(s))])
    # projection coordinates system solving
    lbda = np.matmul(Gram_inv, s)
    sigma = lbda[0] * np.eye(2)
    xi = lbda[1] * np.array([[0, -1], [1, 0]])
    u = np.array([lbda[2], lbda[3]])
    # Infinitesimal similarity
    dG = np.hstack((sigma + xi, u[:, np.newaxis]))
    dG = np.vstack((dG, [0, 0, 0]))
    return dG


def similarity_projection2d(A: GridForm2D, v, heaviside_eps):
    grid = A.grid
    X = grid.X
    Y = grid.Y
    # Similarity space decomposition on the grid
    # B:(4,n,m,2) v:(n,m,2)
    B = np.array([np.dstack((X, Y)),
                  np.dstack((-Y, X))])
    # vectors scalar product (4,n,m)
    s = np.einsum('ijlm,kijml->kij', v[:, :, np.newaxis, :], B[:, :, :, :, np.newaxis])
    p = soft_delta(-A.get_sdf(), heaviside_eps)
    tau = grid.integrate(p * s[0]) * np.array([[1, 0], [0, 1]])
    zeta = grid.integrate(p * s[1]) * np.array([[0, -1], [1, 0]])
    omega = grid.integrate(p[:, :, np.newaxis] * v)

    dG = np.hstack((tau + zeta, omega[:, np.newaxis]))
    dG = np.vstack((dG, [0, 0, 0]))
    return dG



def step(A: GridForm2D, B: GridForm2D, heaviside_eps=0.5, sdf_iter=100, descent_step=0.1, drawers=None):
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
    deform_field = optimal_vector_field(A, B, heaviside_eps, sdf_iter, drawers)
    np.save('v', deform_field)
    chi_A, chi_B = chi_funcs(A, B, heaviside_eps, sdf_iter)
    # Projecting the vector field on the similarities space
    dG = similarity_projection2d(A, deform_field, heaviside_eps)
    G = expm(descent_step * dG)
    # Computing the transformed shape
    GA = A.similarity(G)
    GA = A.deform(deform_field)
    print('Sym diff : ' + str(sym_diff(A.grid, chi_A, chi_B)))
    return GA, G


def optimise(grid: Grid2D, A: GridForm2D, B: GridForm2D, nb_iter, heaviside_eps=0.5, sdf_iter=100, descent_step=0.2, deform=False):
    if deform:
        return optimise_deform(grid, A, B, nb_iter, heaviside_eps, sdf_iter, descent_step)
    else:
        return optimise_no_deform(grid, A, B, nb_iter, heaviside_eps, sdf_iter, descent_step)


def optimise_no_deform(grid: Grid2D, A: GridForm2D, B: GridForm2D, nb_iter, heaviside_eps=0.5, sdf_iter=100, descent_step=0.2):
    fig = plt.figure(figsize=(16, 8))
    c_drawer = ContourDrawer(grid, fig, 141, "Shapes contours")
    sim_drawer = SimDrawer(grid, fig, 143, "Local similarity vector field")
    G_drawer = SimDrawer(grid, fig, 144, "Global similarity vector field")
    vf_drawer = VectorFieldDrawer(grid, fig, 142, "Deformation vector field")
    G = np.eye(3)
    GA = A
    for i in range(nb_iter):
        fig.suptitle('Iteration ' + str(i))
        GA = A.similarity(G)
        c_drawer.draw(GA, B)
        GA, Gi = step(GA, B, heaviside_eps, sdf_iter, descent_step, [vf_drawer])
        G = np.matmul(Gi, G)
        sim_drawer.draw_on_shape(Gi, GA)
        G_drawer.draw_on_shape(G, A)
    plt.ioff()
    plt.show()
    return GA, G


def optimise_deform(grid: Grid2D, A: GridForm2D, B: GridForm2D, nb_iter, heaviside_eps=0.5, sdf_iter=100, descent_step=0.2):
    fig = plt.figure(figsize=(16, 8))
    c_drawer = ContourDrawer(grid, fig, 221, "Shapes contours")
    vf_drawer = VectorFieldDrawer(grid, fig, 222, "Deformation vector field")
    for i in range(nb_iter):
        fig.suptitle('Iteration ' + str(i))
        A = A.deform(optimal_vector_field(A, B, heaviside_eps, sdf_iter, [vf_drawer]), descent_step)
        c_drawer.draw(A, B)
    plt.ioff()
    plt.show()
    return A














