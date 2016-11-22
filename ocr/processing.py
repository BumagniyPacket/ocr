# -*- coding: utf-8 -*-

import numpy as np

from scipy.ndimage.filters import median_filter


def text_detect(image):
    tmp = image.copy()

    tmp = remove_background(tmp)

    return tmp


def remove_background(image):
    """
    Denoising picture function.
        1) Allocate 'background' color by a median filter.
        2) Compute 'foreground' mask as anything that is significantly darker
           than the background.
        3) Return the input value for all pixels in the mask or pure white
           otherwise.

    More information:
        https://www.kaggle.com/rdokov/denoising-dirty-documents/background-removal/files

    :param image: image in np.ndarray type
    :return: type: np.ndarray
    """
    bg = median_filter(image, 11)
    mask = image < bg - 0.1
    return np.where(mask, image, 1.0)
