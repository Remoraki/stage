import initialisation as init
import optimisation as opt
import neural_optimisation as nopt
from utils import save_current, load_current
from neural_sdf.sdf2d import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def A_shape(X, Y):
    return np.logical_and(np.abs(X) < 1, np.abs(Y) < 1).astype(float)

def B_shape(X, Y):
    return np.logical_and(X + 0.2 < 0.5 * (Y + 0 / 5), (X + 0.2) ** 2 < 0.1).astype(float)


def C_shape(X, Y):
    return np.logical_and(np.abs(X) < 2, np.abs(Y) < 2).astype(float)

def align_shapes():
    _, A0, B = init.create_grid_and_functions(A_shape, C_shape, 3, (100, 100))
    alpha = np.pi / 6
    G = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0]])
    B = B.similarity(G)
    grid, A, B = init.initialise(A0, B, align=False)
    GA, G = opt.optimise(grid, A, B, nb_iter=100, descent_step=0.5, heaviside_eps=0.5, sdf_iter=100, deform=False)
    save_current(GA, B)

def align_shapes_neural(compute_sdf=True):
    if compute_sdf:
        _, A0, B = init.create_grid_and_functions(A_shape, C_shape, 3, (100, 100))
        alpha = np.pi / 6
        G = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0]])
        B = B.similarity(G)
        grid, A, B = init.initialise(A0, B, align=False)
        phi_A = torch.from_numpy(A.get_sdf().flatten()[:, np.newaxis]).to(device)
        phi_B = torch.from_numpy(B.get_sdf().flatten()[:, np.newaxis]).to(device)
        A_sdf = train_sdf(grid, phi_A, epochs=500, drawer=SdfDrawer(grid))
        B_sdf = train_sdf(grid, phi_B, epochs=500, drawer=SdfDrawer(grid))
        torch.save(A_sdf, 'Data/A_sdf.pth')
        torch.save(B_sdf, 'Data/B_sdf.pth')
        save_current(A, B)
    else:
        A_sdf = torch.load('Data/A_sdf.pth')
        B_sdf = torch.load('Data/B_sdf.pth')
        grid, _, _ = load_current(True)
        plt.ioff()
        plt.show()
    optimizer = nopt.NeuralOptimiser(grid, device)
    G = optimizer.optimise(A_sdf, B_sdf, 100)
    return A_sdf, B_sdf, grid


if __name__ == '__main__':
    A_sdf, B_sdf, grid = align_shapes_neural(False)
    #align_shapes()



