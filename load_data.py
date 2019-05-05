from keras.datasets import fashion_mnist
import numpy as np
import random


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
catted_cache = None
test_cache = None
test_x_cache = None
test_y_cache = None

def get_train_data(mode="mlp"):
    global catted_cache
    if catted_cache is None:
        catted_cache = x_train
        catted_cache = catted_cache / 127.5 - 1.

    if mode == "cnn":
        return catted_cache
    if mode == "mlp":
        return catted_cache.reshape(60_000, 784)


def gen_test_data(num_pts=1024, mode='mlp'):
    global test_x_cache, test_y_cache, x_test
    train_data = get_train_data()
    train_random_idxs = list(range(x_train.shape[0]))
    test_random_idxs = list(range(x_test.shape[0]))
    random.shuffle(train_random_idxs)
    random.shuffle(test_random_idxs)
    train_random_idxs = train_random_idxs[:num_pts // 2]
    test_random_idxs = test_random_idxs[:num_pts - num_pts // 2]
    test_x_cache = np.empty((num_pts,) + catted_cache.shape[1:])
    test_y_cache = np.empty((num_pts,))
    
    test_x_cache[range(len(train_random_idxs))] = train_data[train_random_idxs]
    test_y_cache[range(len(train_random_idxs))] = 0  # 0 means not anomalous

    x_test = x_test / 127.5 - 1.

    test_x_cache[range(len(test_random_idxs), num_pts)] = x_test[test_random_idxs]
    test_y_cache[range(len(test_random_idxs), num_pts)] = 1  # 1 means anomalous
    
    return test_x_cache, test_y_cache


def get_test_data():
    if test_cache is None:
        test_cache = catted_cache

        for i in range(len(x_test)):
            random_index = random.randint(0, len(test_cache))
            test_cache = np.insert(test_cache, random_index, x_test[i], axis = 0)

    return test_cache
