import numpy as np
import torch
from sdf2d import *
import matplotlib.pyplot as plt
from redistance import redistance
from grids import *
import utils as utl

# TODO: check whether cuda is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TODO: create a simple sdf for a circle, from which we can sample points
# TODO: setup training procedure


def shape(grid: Grid2D):
    X = grid.X
    Y = grid.Y
    chi = np.logical_and(np.abs(X) < 0.5, np.abs(Y) < 0.5)
    return grid.form(chi.astype(float))


if __name__ == '__main__':
    grid = Grid2D((-1, -1), (1, 1), (100, 100))
    form = shape(grid)
    true_sdf = torch.from_numpy(form.get_sdf().flatten()[:, np.newaxis]).float().to(device)
    SdfDrawer(grid).draw(true_sdf.cpu().numpy())
    trained_sdf = train_sdf(grid, true_sdf, drawer=SdfDrawer(grid), weight=0.1, epochs=500)

    plt.ioff()
    plt.show()

