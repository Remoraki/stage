from grids import *
import numpy as np
import matplotlib.pyplot as plt



def save(grid, A, B):
    np.save('B', B.chi)
    np.save('A', A.chi)
    grid_dic = {'res': grid.resolution, 'bl': grid.bottom_left, 'tr': grid.top_right}
    np.save('grid', grid_dic, True)


def load():
    grid_info = np.load('grid.npy', allow_pickle=True).item()
    grid = Grid2D(grid_info['bl'], grid_info['tr'], grid_info['res'])
    A = grid.form(np.load('A.npy'))
    B = grid.form(np.load('B.npy'))
    return grid, A, B



def plot(form: GridForm2D, name='form', scaled=False):
    fig = plt.figure()
    fig.suptitle(name)
    ax0 = plt.subplot(121, projection='3d')
    ax1 = plt.subplot(122, projection='3d')
    if scaled:
        f, _  = form.rescale()
    else:
        f = form
    f.plot_surface(ax0, title='Chi')
    f.plot_sdf(ax1, title='Phi')


if __name__ == '__main__':
    grid, A, B = load()
    _, [A, B] = rescale_all([A, B], square_output=True)
    plot(A, 'A')
    plot(B, 'B')
    plt.show()
