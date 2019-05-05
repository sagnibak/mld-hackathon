from keras.datasets import fashion_mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
catted_cache = None

def get_train_data():
    if catted_cache is None:
        catted_cache = np.concatenate((x_train, x_test), axis=0)
    return catted_cache


def get_test_data():
    raise NotImplementedError("Need to generate testing data somehow")
