from keras.datasets import fashion_mnist
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
catted_cache = None

def get_train_data(mode="mlp"):
    if catted_cache is None:
        catted_cache = x_train
        catted_cache = catted_cache / 127.5 - 1.

    if mode == "cnn":
        return catted_cache
    if mode == "mlp":
        return catted_cache.reshape(60000, 784)


def get_test_data():
    train_data = get_train_data()
    test_data = x_test

    num_train, num_test, num_total = 0, 0, 0
    selected_train, selected_test = set(), set()

    test_cache = np.empty([10000, 784])

    normal_data_indexes, normal_index = np.empty(5000), 0

    while num_train < 5000 and num_test < 5000:
        if random.random() < 0.5:
            randomIndex = random.randint(0, len(train_data) - 1)
            if not randomIndex in selected_train:
                for i in range(len(train_data[randomIndex])):
                    test_cache[num_total][i] = train_data[randomIndex][i]
                num_total += 1
                num_train += 1
                selected_train.add(randomIndex)
                normal_data_indexes[normal_index] = num_total
        else:
            randomIndex = random.randint(0, len(test_data) - 1)
            if not randomIndex in selected_test:
                for i in range(len(test_data[randomIndex].flatten())):
                    test_cache[num_total][i] = test_data[randomIndex].flatten()[i]
                num_total += 1
                num_test += 1
                selected_test.add(randomIndex)

    while num_train < 5000:
        randomIndex = random.randint(0, len(train_data) - 1)
        if not randomIndex in selected_train:
            for i in range(len(train_data[randomIndex])):
                test_cache[num_total][i] = train_data[randomIndex][i]
            num_total += 1
            num_train += 1
            selected_train.add(randomIndex)
            normal_data_indexes[normal_index] = num_total

    while num_test < 5000:
        randomIndex = random.randint(0, len(test_data) - 1)
        if not randomIndex in selected_test:
            for i in range(len(test_data[randomIndex].flatten())):
                test_cache[num_total][i] = test_data[randomIndex].flatten()[i]
            num_total += 1
            num_test += 1
            selected_test.add(randomIndex)

    return test_cache, normal_data_indexes
