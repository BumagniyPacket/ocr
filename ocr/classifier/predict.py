from sklearn.externals import joblib
from skimage.transform import resize
from config import CLASSIFIER_PATH

clf = joblib.load(CLASSIFIER_PATH)


def predict(character):
    # char = allocate_letter(character)
    char = resize(character, (20, 20)).ravel().reshape(1, -1)

    liter_code = int(clf.predict(char))
    liter = chr(liter_code)

    return liter
