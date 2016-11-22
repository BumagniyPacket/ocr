from sklearn.externals import joblib
from skimage.transform import resize
from ocr.segmentation import allocate_letter
from config import CLASSIFIER_PATH

clf = joblib.load(CLASSIFIER_PATH)


def predict(character):
    char = allocate_letter(character)
    char = resize(char, (20, 20))

    liter_code = int(clf.predict(char.ravel().tolist()))
    liter = chr(liter_code)

    return liter
