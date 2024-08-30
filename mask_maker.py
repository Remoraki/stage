from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

white = [0, 0, 0, 0]

masks = ['fish1', 'fish2']

for mask_name in masks:
    image = np.array(Image.open('Data/' + mask_name + '.png'))
    mask = np.logical_not(np.all(image == white, axis=-1))
    np.save('Data/' + mask_name + '_mask.npy', mask)
    plt.imshow(image)
    plt.show()
