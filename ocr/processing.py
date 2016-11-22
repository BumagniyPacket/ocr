# -*- coding: utf-8 -*-
from scipy import signal

import numpy as np
from skimage.filters.rank import mean_bilateral
from skimage.filters.rank import mean_percentile
from skimage.measure import approximate_polygon

from ocr import paint
from skimage.feature import hog as _hog
from skimage.filters import threshold_adaptive, canny
from scipy.ndimage.filters import maximum_filter, median_filter, minimum_filter
from skimage.filters.rank import mean_bilateral as mean
from skimage.filters.rank import gradient
from skimage.filters import laplace
from skimage.exposure import equalize_adapthist
from skimage.morphology import disk
from skimage.transform import rescale, resize
from scipy.signal import convolve2d as conv
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, erosion
from skimage.transform import hough_line, hough_line_peaks, probabilistic_hough_line
from skimage.draw import line
from skimage.measure import find_contours
from matplotlib import pyplot as plt
from skimage.filters import gaussian_filter
from scipy.signal import medfilt2d


def text_detect(image):
    tmp = image.copy()

    # print(tmp)
    # print(mean(tmp, np.ones(2, 2)))

    # tmp = contrast_up(tmp, 5)

    tmp = remove_background(tmp)
    # tmp = angle(tmp)

    return tmp


def contrast_up(image, seed_size=10, k=6):
    map_ = map_maker(image)

    ret = k * (image - map_) + map_

    return ret


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


def pyramid_decomposition(image):
    """
     Разложение начинается с масштаба исходного изображения. Оно делится на
     непересекающиеся квадраты размером 2х2 пикселя, в каждом из которых мы
     получаем значения минимума, максимума и среднего из 4-х пикселей, его
     составляющих. Далее из этих значений формируем три изображения:
     минимумов, максимумов и средних, которые уменьшены в 2 раза по
     горизонтали и вертикали относительно исходного. Повторяем процедуру и
     раскладываем полученные изображения в пирамиды до уровня, на котором
     размер ещё составляет не менее 2 пикселей по горизонтали и вертикали.
    :param image:
    :return:
    """
    width, height = image.shape
    factor = 2
    max_list, min_list, mean_list = [], [], []

    image_2_map = image.copy()

    map_max = maximum_filter(image_2_map, factor)
    map_min = minimum_filter(image_2_map, factor)
    map_mean = mean_percentile(image_2_map, np.ones((2, 2))) / 255.

    while min(width, height) > 3:

        width, height = int(width / factor), int(height / factor)

        map_max = resize(map_max, (width, height))
        map_min = resize(map_min, (width, height))
        map_mean = resize(map_mean, (width, height))

        width, height = map_max.shape

        max_list.append(map_max)
        min_list.append(map_min)
        mean_list.append(map_mean)

        map_max = maximum_filter(map_max, factor)
        map_min = minimum_filter(map_min, factor)
        map_mean = mean(map_mean, np.ones((2, 2)))

    return max_list, min_list, mean_list
