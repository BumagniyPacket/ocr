import numpy as np
from skimage.filters import gaussian_filter
from skimage.filters import threshold_otsu
from skimage.measure import find_contours


def detect_parts(img, level=10):
    gauss = gaussian_filter(img, level)
    thresh = threshold_otsu(gauss)
    return gauss < thresh


# TODO: normal'nuu segmentaciu delai
def segmentation(img):
    contours = find_contours(detect_parts(img, 10), .8)
    return [img[min(c[:, 0]):max(c[:, 0]), min(c[:, 1]):max(c[:, 1])] for c in contours]


def line_segmentation(segment, level=1, dst=None):
    level = level
    lines = []
    binary = segment > .7

    up, down = 0, 0
    for n, bright in enumerate([np.mean(line) for line in binary]):
        # TODO: vmesto cifri odin v uslovii nugen adaprovanniy pod segment porog(v ideale konechno nugen porog=1)
        if bright < level and not up:
            up = n - 1
            down = 0
        if bright >= level and not down and up:
            down = n
            lines.append(binary[up:down, 0:segment.shape[1]])
            up = 0

    if dst is None:
        return lines
    else:
        dst += [_ for _ in lines]


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
            liters.append((left_w, n - left_w, spaces[-1]))
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
