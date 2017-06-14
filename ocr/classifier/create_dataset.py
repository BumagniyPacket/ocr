import os
import numpy as np

from skimage import io

PATH = 'data'

os.listdir(PATH)

with open('dataset.csv', 'w') as dataset:
    for filename in os.listdir(PATH):
        if filename == '.directory':
            continue

        tmp_img = io.imread(os.path.join(PATH, filename), as_grey=True) > .5
        image = ','.join(np.asarray(tmp_img.ravel() * 1, dtype=str))
        label = ord(filename[0])
        line = '%s,%s\n' % (label, image)

        dataset.writelines(line)

print('Done!')
