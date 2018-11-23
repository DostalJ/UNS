import pickle
import numpy as np
import tensorflow as tf

# toto je jen skareda droobnost, ktera nas zbavi otravne varujici hlasky
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = './data/MNIST.pkl'
with open(file=PATH, mode='rb') as pkl_file:
    save = pickle.load(file=pkl_file)
    X_train = save['X_train']
    y_train = save['y_train']
    X_valid = save['X_valid']  # nacteme i validacni data
    y_valid = save['y_valid']  # nacteme i validacni data
    X_test = save['X_test']  # nacteme i tesotvaci data
    y_test = save['y_test']  # nacteme i tesotvaci data

    del save

print('#'*40)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_valid:", X_valid.shape)  # pridame i validacni data
print("Shape of y_valid:", y_valid.shape)  # pridame i validacni data
print("Shape of X_test:", X_test.shape)  # pridame i testovaci data
print("Shape of y_test:", y_test.shape)  # pridame i testovaci data
print('#'*40)


# ###############################
# HYPERPARAMETERS
num_features = X_train.shape[1]
num_labels = y_train.shape[1]

UNITS_HL = 1024  # pocet neuronu ve skryte vrstve
# ###############################

graph = tf.Graph()
with graph.as_default():
    X_train_tf = tf.constant(X_train)
    y_train_tf = tf.constant(y_train)

    X_valid_tf = tf.constant(X_valid)  # pridejme validacni data do TF grafu
    X_test_tf = tf.constant(X_test)  # pridejme testovaci data do TF grafu

    # nove vahy a biases
    w1 = tf.Variable(tf.truncated_normal(shape=[num_features, UNITS_HL]))
    b1 = tf.Variable(tf.zeros(shape=[UNITS_HL]))
    w2 = tf.Variable(tf.truncated_normal(shape=[UNITS_HL, num_labels]))
    b2 = tf.Variable(tf.zeros(shape=[num_labels]))

    # pro prehlednost definujeme vypocet v grafu jako funkci
    def feed_forward(X):
        layer1 = tf.matmul(X, w1) + b1  # prvni vrstva
        layer1 = tf.nn.relu(layer1)  # akrivace prvni vrstvy
        out_layer = tf.matmul(layer1, w2) + b2  # vystupni vrstva (bez aktivace)
        return out_layer  # vratime vystup (bez aktivace!!)

    logits = feed_forward(X_train_tf)  # neaktivovany vystup z treninkovych dat
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y_train_tf))

    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    train_pred = tf.nn.softmax(logits)  # predikci vytvorime aktivaci vystupu

    # predikujeme take z validacnich a testovacich dat
    valid_pred = tf.nn.softmax(feed_forward(X_valid_tf))
    test_pred = tf.nn.softmax(feed_forward(X_test_tf))


def accuracy(pred, labels):
    acc = (np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])
    return acc*100


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized!")
    for step in range(250):
        _, l, pred = session.run([optimizer, loss, train_pred])
        if step % 25 == 0:
            print("Loss at step {:d}: {:.4f}".format(step, l))
            print("Training accuracy: {:.1f} %".format(
                accuracy(pred, y_train)))
            # validacni uspesnost 
            print('Validation accuracy: {:.1f}%'.format(
                accuracy(valid_pred.eval(), y_valid)))
            print('-'*20)
    # testovaci uspesnost
    print('Test accuracy: {:.1f}%'.format(accuracy(test_pred.eval(), y_test)))
