from skimage import color
from skimage import io
from skimage.transform import rescale
from config import MAX_SIZE


def downscale(image):
    """
    Если одна из сторон больше MAX_SIZE то уменьшаем в (MAX_SIZE / max_shape) раз
    :param image:
    :return:
    """
    max_shape = max(image.shape)
    if max_shape > MAX_SIZE:
        scale = MAX_SIZE / max_shape
        return rescale(image, scale)
    else:
        return image


def img_open(path):
    """
    Открытие файла в градациях серого
    :param path: путь к обрабатываемому изображению
    :return: изображение
    """
    image = color.rgb2gray(io.imread(path) / 255.0)
    return downscale(image)


def img_write(image, path='ret.png'):
    """

    :param image:
    :param path:
    :return:
    """
    io.imsave(path, image)
