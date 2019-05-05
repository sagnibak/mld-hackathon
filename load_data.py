from keras.datasets import fashion_mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
catted_cache = None

def get_train_data(mode="mlp"):
    if catted_cache is None:
        catted_cache = np.concatenate((x_train, x_test), axis=0
                        ).astype(np.float32)
        catted_cache = catted_cache / 127.5 - 1.

    if mode == "cnn":
        return catted_cache
    if mode == "mlp":
        return catted_cache.reshape(70_000, 784)


def get_test_data():
    raise NotImplementedError("Need to generate testing data somehow")
