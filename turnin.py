from keras.models import load_model
import numpy as np

r = -9.611524001229554e-05

final_model = load_model('final_model.h5', compile=False)

decision_score = lambda img: final_model.predict(img) - r
decision = lambda d_score: int(d_score < 0)


def preprocess(img):
    """Takes in an array of shape (N, 28, 28) containing values
    in the range [0, 255], and returns a flattened array of float32
    with shape (N, 784) and values in the range [-1, 1].
    """
    img = img.astype(np.float32)
    img = img / 127.5 - 1.
    return img.reshape(-1, 784)


def predict(img):
    return decision(decision_score(preprocess(img)))
