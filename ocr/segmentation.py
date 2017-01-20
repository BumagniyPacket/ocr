import numpy as np
from skimage.filters import sobel
from skimage.measure import find_contours
from skimage.morphology import binary_closing, binary_opening, dilation
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
    image = ~(image > .5)

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
    # список для хранения строк
    lines = []
    # коэфициент для разделения строк
    c = 0
    # ширина сегмента
    width = segment.shape[1]

    up = down = 0

    brights = [np.mean(line) for line in segment]
    for n, bright in enumerate(brights):
        if bright > c and not up:
            up = n - 1
            down = 0
        if bright <= c and not down and up:
            down = n
            lines.append(segment[up:down, 0:width])
            up = 0

    return lines


def symbol_segmentation(line):
    """

    :param line:
    :return:
    """
    # ширина строки
    line_width = line.shape[1]
    # cредняя яркость столбов пикселей фрагмента строки
    mean_bright_l = [np.mean(line[:, __]) for __ in range(line_width)]
    # коэффициент по которому будем разделять слова и символы
    c = 0
    # временные координаты для выделения символа и пробела
    space_l = space_r = symbol_l = symbol_r = 0
    # координаты пробела
    space = None
    # акумулятор среднего значения пробелов
    mean_space = 0
    # список для хранения символов, список на возврат
    liters, ret_words = [], []

    for n, bright in enumerate(mean_bright_l):
        if not space_l and bright <= c:
            space_l, space_r = n, 0
        elif space_l and not space_r and bright > c:
            space = n - space_l
            mean_space += space
            space_l, space_r = 0, n

        if not symbol_l and not space_l and bright > c:
            symbol_l, symbol_r = n, 0
        elif symbol_l and not symbol_r and bright <= c:
            # (start of word, symbol width, space length before liter)
            liters.append((symbol_l, n - symbol_l, space))
            symbol_l, symbol_r = 0, n

    mean_space /= len(liters)

    for lit in liters:
        start, end, space = lit
        if not space < mean_space:
            ret_words.append(' ')
        symbol = line[0:line_width, start:start + end]
        ret_words.append(allocate_symbol(symbol))

    return ret_words + ['\n']


def allocate_symbol(symbol):
    """

    :param symbol:
    :return:
    """
    #
    mean = [np.mean(_) for _ in symbol]
    #
    up, down = 0, len(symbol) - 1
    #
    с = 0

    while mean[up] <= с:
        up += 1
    while mean[down] <= с:
        down -= 1
    return symbol[up - 1:down + 1, :]

