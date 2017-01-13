import numpy as np
from skimage.filters import gaussian
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import binary_closing, binary_opening, dilation
from skimage.measure import find_contours
from skimage.transform import rescale


def check_intersection(segments):
    def check_xy(a11, a12, a21, a22):
        return a21 <= (a11 + a12) / 2 <= a22

    for s_checking in sorted(segments, key=lambda x: x[0].size):
        for s in segments:

            s1 = s_checking[1]
            s2 = s[1]

            if s1 == s2:
                continue

            if check_xy(s1[0], s1[1], s2[0], s2[1]) and \
                    check_xy(s1[2], s1[3], s2[2], s2[3]):
                # TODO: nutiznaesh
                try:
                    segments.remove(s_checking)
                except ValueError:
                    pass

    return segments


def segments_extraction(image):
    image = np.invert(image > .5)

    w, h = image.shape

    scale = 800 / max(w, h)
    scaled = rescale(image, scale)

    w, h = scaled.shape

    window_o = np.ones((1, int(w / 100)))
    window = np.ones((8, 8))

    sobeled = sobel(scaled)
    open_image = binary_closing(sobeled, window_o)
    close_image = binary_opening(open_image, window_o)

    dilate = dilation(close_image, window)

    contours = find_contours(dilate, .8)

    segments = []
    for contour in contours:

        segment = image[min(contour[:, 0]) / scale:max(contour[:, 0]) / scale,
                        min(contour[:, 1]) / scale:max(contour[:, 1]) / scale]
        if segment.mean() <= .05:
            continue

        coords = min(contour[:, 0]) / scale, \
                 max(contour[:, 0]) / scale, \
                 min(contour[:, 1]) / scale, \
                 max(contour[:, 1]) / scale
        segments.append((segment, coords))

    return segments


def segmentation(img):
    segments = segments_extraction(img)
    return check_intersection(segments)


def line_segmentation(segment):
    """
    В каждой строке массива сегмента расчитывается среднее значение интенсивности пикселей.
    Если это среднее меньше 1.(самое большое значение интенсивности) значит полезная информация в данной строке есть.
    Пока значение среднего не будет ровнятся 1. строки записываются...
    :param segment:
    :return:
    """
    lines = []
    level = 0
    width = segment.shape[1]

    up = down = 0
    brights = [np.mean(line) for line in segment]
    for n, bright in enumerate(brights):
        if bright > level and not up:
            up = n - 1
            down = 0
        if bright <= level and not down and up:
            down = n
            lines.append(segment[up:down, 0:width])
            up = 0

    return lines


def word_segmentation(line, level=1):
    pixel_bright = [np.mean(line[:, ax]) for ax in range(line.shape[1])]
    left_s = right_s = left_w = right_w = 0
    spaces, liters, ret_words = [], [], []

    for n, bright in enumerate(pixel_bright):
        if not left_s and bright >= level:
            left_s, right_s = n, 0
        elif left_s and not right_s and bright < level:
            spaces.append(n - left_s)
            left_s, right_s = 0, n

        if not left_w and not left_s and bright < level:
            left_w, right_w = n, 0
        elif left_w and not right_w and bright >= level:
            # liters.append((start of word, liter length,
            #                space length before liter))
            try:
                liters.append((left_w, n - left_w, spaces[-1]))
            except IndexError:
                pass
            left_w, right_w = 0, n

    mean_space = np.mean(spaces)

    for lit in liters:
        if not lit[2] < mean_space:
            ret_words.append(' ')
        litter = line[0:line.shape[1], lit[0]:lit[0] + lit[1]]
        ret_words.append(litter)

    return ret_words + ['\n']


def allocate_letter(letter):
    mean = [np.mean(_) for _ in letter]
    center = mean.index(min(mean))

    up, down = center, center
    for x in range(center, 0, -1):
        if mean[x] == 1:
            break
        up = x
    for x in range(center, len(mean)):
        if mean[x] == 1:
            break
        down = x + 1
    return letter[up:down, 0:letter.shape[1]] * 1.
