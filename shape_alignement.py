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


def A_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return np.logical_and(np.abs(X) < 1, np.abs(Y) < 1).astype(float)

def B_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return (X**2 + Y**2 <= 3).astype(float)

def C_shape(grid: Grid2D):
    X, Y = grid.X, grid.Y
    return np.logical_and(np.abs(X) < 0.8, np.abs(Y) < 2.2).astype(float)

def image_shape(grid: Grid2D, mask_path):
    om = np.load(mask_path)
    shape = (om.shape[1], om.shape[0])
    if shape != grid.shape:
        if grid.size[0] > grid.size[1]:
            xSize = grid.size[0]
            ySize = xSize * shape[1] / shape[0]
        else:
            ySize = grid.size[1]
            xSize = ySize * shape[0] / shape[1]
        bl = (-xSize / 2, -ySize / 2)
        tr = (xSize / 2, ySize / 2)
        ngrid = Grid2D(bl, tr, shape)
        form = ngrid.form(om)
        nform = grid.form_interpolation(form.flatten())
        np.save(mask_path, nform.chi)
    else:
        nform = grid.form(om)
    return nform.chi

def image_shape_A(grid):
    return image_shape(grid, 'Data/fish1_mask.npy')

def image_shape_B(grid):
    return image_shape(grid, 'Data/fish2_mask.npy')

def square_sdf_image(P, center, size):
    half_size = size / 2
    shifted_P = torch.abs(P - center) - half_size
    outside_distance = torch.max(shifted_P, torch.tensor(0.0).to(P.device))
    inside_distance = torch.min(torch.max(shifted_P, dim=1).values, torch.tensor(0.0).to(P.device))
    sdf = outside_distance.norm(dim=1) + inside_distance
    return sdf[:, torch.newaxis]

def align_shapes():
    _, A0, B = init.create_grid_and_functions(A_shape, C_shape, 3, (100, 100))
    alpha = np.pi / 6
    G = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0]])
    B = B.similarity(G)
    grid, A, B = init.initialise(A0, B, align=False)
    GA, G = opt.optimise(grid, A, B, nb_iter=100, descent_step=0.5, heaviside_eps=0.5, sdf_iter=100, deform=False)
    save_current(GA, B)

def align_shapes_neural(compute_sdf=True, compute_grids=True):
    if compute_grids:
        print('Creating shapes...')
        _, A0, B = init.create_grid_and_functions(image_shape_A, image_shape_B, 3, (300, 300))
        alpha = np.pi / 6
        G = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0]])
        B = B.similarity(G)
        grid, A, B = init.initialise(A0, B, align=False)
        save_current(A, B)
    else:
        grid, A, B = load_current(True)
    if compute_sdf:
        print('Training...')

        phi_A = torch.from_numpy(A.get_sdf(300).flatten()[:, np.newaxis]).to(device)
        A_scaling = torch.max(torch.abs(phi_A)).item()
        np.save('Data/A_scaling.npy', A_scaling)
        phi_A /= A_scaling
        plt.imshow(np.reshape(phi_A.cpu().numpy(), grid.shape))
        #phi_A = square_sdf_image(grid.tensor(device), torch.tensor([0, 0]).to(device), 2)
        phi_B = torch.from_numpy(B.get_sdf(300).flatten()[:, np.newaxis]).to(device)
        B_scaling = torch.max(torch.abs(phi_B)).item()
        np.save('Data/B_scaling.npy', B_scaling)
        phi_B /= B_scaling

        fig = plt.figure()
        A_sdf = train_sdf(grid, phi_A, epochs=200, drawer=SdfDrawer(grid, fig, 111))
        torch.save(A_sdf, 'Data/A_sdf.pth')
        B_sdf = train_sdf(grid, phi_B, epochs=200, drawer=SdfDrawer(grid, fig, 111))
        torch.save(B_sdf, 'Data/B_sdf.pth')
    else:
        A_sdf = torch.load('Data/A_sdf.pth')
        A_scaling = np.load('Data/A_scaling.npy').item()
        B_sdf = torch.load('Data/B_sdf.pth')
        B_scaling = np.load('Data/B_scaling.npy').item()
        plt.ioff()
        plt.show()
        print('Optimizing...')
    optimizer = nopt.NeuralOptimiser(grid, device)
    G = optimizer.optimise(A_sdf, B_sdf, A_scaling, B_scaling, 10)
    return A_sdf, B_sdf, grid, G


def drawA():
    grid, A, B = load_current(True)
    A_sdf = torch.load('Data/A_sdf.pth')
    fig = plt.figure()
    SdfDrawer(grid, fig, 111).draw_sdf(A_sdf, device)




if __name__ == '__main__':
    grid = Grid2D((-1,-1), (1, 1), (100, 100))

    A_net = SDFNet()
    B_net = SDFNet()
    A_dict = torch.load('Data/save.pth')
    B_dict = torch.load('Data/square.pth')
    A_net.load_state_dict(A_dict)
    B_net.load_state_dict(B_dict)
    A_net.to(device)
    B_net.to(device)
    initializer = ninit.NeuralInitializer(grid, device, A_net, B_net)
    optimiser = nopt.NeuralOptimiser(grid, device, initializer)
    optimiser.optimise()
    np.save('Data/G.npy', optimiser.G)



