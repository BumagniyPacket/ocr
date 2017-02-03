from ocr.classifier.predict import predict
from ocr.io import *
from ocr.processing import *
from ocr.segmentation import *


def get_text(filename):
    ret_string = ''
    # открываем
    image = img_open(filename)
    # чистим
    clear_image = text_detect(image)
    # сегментируем
    segments = segmentation(clear_image)

    for segment in segments:
        lines = line_segmentation(segment)
        symbols = []

        segment_symbol_mean_height = 0

        for line in lines:
            if type(line) == str:
                symbols += line
                continue
            tmp = symbol_segmentation(line)
            symbols += tmp[0]
            # средняя высота по строке
            segment_symbol_mean_height += tmp[1] / len(tmp[0])

        symbols += ['\n']
        # средняя высота по сегменту
        height = segment_symbol_mean_height / len(lines)

        for symbol in symbols:
            if not type(symbol) == str:
                h, w = symbol.shape
                good_symbol = predict(symbol)

                if h < (height + 1):
                    good_symbol = good_symbol.lower()
                if h < height / 2:
                    good_symbol = '.'

                ret_string += good_symbol
            else:
                ret_string += symbol

    return ret_string


def magic(filename):
    try:
        return get_text(filename)
    except:
        return 'Text not found!'
