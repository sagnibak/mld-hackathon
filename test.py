import tensorflow as tf
import time
from keras.datasets import fashion_mnist
import numpy as np
import random

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def get_train_data(mode="mlp"):
    catted_cache = x_train / 127.5 - 1

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

def tf_mnist_OneClass_NN_linear(data_train,data_test):

    print("Started function")

    tf.reset_default_graph()

    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)

    train_X = data_train

     # Layer's sizes
    x_size = train_X.shape[1]   # Number of input nodes: 4 features and 1 bias
    h_size = 32                # Number of hidden nodes
    y_size = 1   # Number of outcomes (3 iris flowers)
    D = x_size
    K = h_size

    theta = np.random.normal(0, 1, K + K*D + 1)
    rvalue = np.random.normal(0,1,(len(train_X),y_size))
    nu = 0.04


    def init_weights(shape):
        """ Weight initialization """
        weights = tf.random_normal(shape,mean=0, stddev=0.00001)
        return tf.Variable(weights)

    def forwardprop(X, w_1, w_2):
        """
        Forward-propagation.
        IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
        """
        X = tf.cast(X, tf.float32)
        w_1 = tf.cast(w_1, tf.float32)
        w_2 = tf.cast(w_2, tf.float32)
        h    = (tf.matmul(X, w_1))  #
        yhat = tf.matmul(h, w_2)  # The \varphi function
        return yhat

    g   = lambda x : x

    def nnScore(X, w, V, g):
        X = tf.cast(X, tf.float32)
        w = tf.cast(w, tf.float32)
        V = tf.cast(V, tf.float32)
        return tf.matmul(g((tf.matmul(X, w))), V)

    def relu(x):
        y = x
    #     y[y < 0] = 0
        return y

    def ocnn_obj(theta, X, nu, w1, w2, g,r):

        w = w1
        V = w2


        X = tf.cast(X, tf.float32)
        w = tf.cast(w1, tf.float32)
        V = tf.cast(w2, tf.float32)


        term1 = 0.5  * tf.reduce_sum(w**2)
        term2 = 0.5  * tf.reduce_sum(V**2)
        term3 = 1/nu * tf.reduce_mean(relu(r - nnScore(X, w, V, g)))
        term4 = -r

        return term1 + term2 + term3 + term4




    # For testing the algorithm
    test_X = data_test

    # Symbols
    X = tf.placeholder("float32", shape=[None, x_size])

    r = tf.get_variable("r", dtype=tf.float32,shape=(),trainable=False)

    print("Started weight initialization")

    # Weight initializations
    w_1 = init_weights((x_size, h_size))
    w_2 = init_weights((h_size, y_size))

    print("Started forwardprop")

    # Forward propagation
    yhat    = forwardprop(X, w_1, w_2)
    predict = tf.argmax(yhat, axis=1)

    print("Started training GradientDescentOptimizer")

    # Backward propagation
    # cost    = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=yhat))
    cost    = ocnn_obj(theta, X, nu, w_1, w_2, g,r)
    updates = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

    print("Finished training GradientDescentOptimizer")

    # Run SGD
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    rvalue = 0.1
    start_time = time.time()
    for epoch in range(100):
            # Train with each example
                sess.run(updates, feed_dict={X: train_X,r:rvalue})
                rvalue = nnScore(train_X, w_1, w_2, g)
                with sess.as_default():
                    rvalue = rvalue.eval()
                    rvalue = np.percentile(rvalue,q=100*nu)
                print("Epoch = %d, r = %f"
                  % (epoch + 1,rvalue))

    trainTime = time.time() - start_time

    start_time = time.time()

    train = nnScore(train_X, w_1, w_2, g)
    test = nnScore(test_X, w_1, w_2, g)
    testTime = time.time() - start_time
    with sess.as_default():
        arrayTrain = train.eval()
        arrayTest = test.eval()
    #     rstar = r.eval()

    rstar =rvalue
    sess.close()
    print("Session Closed!!")


    pos_decisionScore = arrayTrain-rstar
    neg_decisionScore = arrayTest-rstar

    write_decisionScores2Csv(decision_scorePath, "OC-NN_RBF.csv.csv", pos_decisionScore, neg_decisionScore)
    return [pos_decisionScore, neg_decisionScore,trainTime,testTime]

train_data, test_data = get_train_data(), get_test_data()
print("Finished getting data")
print(tf_mnist_OneClass_NN_linear(train_data, test_data))
