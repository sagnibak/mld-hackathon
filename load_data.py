from keras.datasets import fashion_mnist
import numpy as np
import random


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
catted_cache = None
test_cache = None

def get_train_data(mode="mlp"):
    global catted_cache
    if catted_cache is None:
        catted_cache = x_train
        catted_cache = catted_cache / 127.5 - 1.

    if mode == "cnn":
        return catted_cache
    if mode == "mlp":
        return catted_cache.reshape(60_000, 784)


def get_test_data():
    if test_cache is None:
        test_cache = catted_cache

        for i in range(len(x_test)):
            random_index = random.randint(0, len(test_cache))
            test_cache = np.insert(test_cache, random_index, x_test[i], axis = 0)

    return test_cache
