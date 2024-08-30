import matplotlib
import numpy as np
import torch

import initialisation as init
import neural_initialisation as ninit
import optimisation as opt
import neural_optimisation as nopt
from utils import save_current, load_current, draw_form_contour, SimDrawer
from neural_sdf.sdf2d import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""Two functions to use : One that uses regular grid and functions that are defined directly on it,
and one that uses sdf networks.
In the examples below, grid functions are explicitly defined, and networks are loaded through weight dictionaries.
The other files in the alignment folder are used throughout the process and don't need to be run"""


# Definitions of literal expressions for shapes to use with grids
def A_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return np.logical_and(np.abs(X) < 1, np.abs(Y) < 1).astype(float)

def B_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return (X**2 + Y**2 <= 3).astype(float)

def C_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return np.logical_and(np.abs(X) < 1, np.abs(Y) < 1).astype(float)

# Alignment functions

def align_shapes():
    """
    Perform the alignment of two shapes defined with grids
    :return:
    """
    _, A0, B = init.create_grid_and_functions(A_shape, C_shape, 3, (100, 100))
    alpha = np.pi / 6
    G = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0]])
    B = B.similarity(G)
    grid, A, B = init.initialise(A0, B, align=False)
    GA, G = opt.optimise(grid, A, B, nb_iter=100, descent_step=5, heaviside_eps=0.5, sdf_iter=100, deform=False)
    save_current(GA, B)

def align_shapes_neural():
    """
    Perform the alignment of two shapes defined with sdf neural networks
    :return:
    """
    grid = Grid2D((-1, -1), (1, 1), (100, 100))
    A_net = SDFNet()
    B_net = SDFNet()
    A_dict = torch.load('Data/save.pth')
    B_dict = torch.load('Data/square.pth')
    A_net.load_state_dict(A_dict)
    B_net.load_state_dict(B_dict)
    A_net.to(device)
    B_net.to(device)
    initializer = ninit.NeuralInitializer(grid, device, A_net, B_net)
    initializer.align()
    optimiser = nopt.NeuralOptimiser(grid, device, initializer)
    optimiser.optimise()
    np.save('Data/G.npy', optimiser.G)



if __name__ == '__main__':
    align_shapes_neural()



