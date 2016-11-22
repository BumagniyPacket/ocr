from ocr.classifier.predict import predict
from ocr.io import *
from ocr.processing import *
from ocr.segmentation import *


# TODO: peredelat' eto ubogestvo
def magic(filename):
    img = img_open(filename)

    dns = text_detect(img)
    lines = []
    [line_segmentation(_, dst=lines) for _ in segmentation(dns)]

    ret_string = ''
    for line in lines:
        letters = word_segmentation(line)
        for lit in letters:
            if type(lit) is str:
                ret_string += lit
            else:
                ret_string += predict(lit)

    return ret_string


