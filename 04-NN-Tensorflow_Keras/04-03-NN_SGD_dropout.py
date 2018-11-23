import pickle
import numpy as np
import tensorflow as tf

from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = './data/MNIST.pkl'
with open(file=PATH, mode='rb') as pkl_file:
    save = pickle.load(file=pkl_file)
    X_train = save['X_train']
    y_train = save['y_train']
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    X_test = save['X_test']
    y_test = save['y_test']

    del save

print('#'*40)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_valid:", X_valid.shape)
print("Shape of y_valid:", y_valid.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print('#'*40)


# ###############################
num_features = X_train.shape[1]
num_labels = y_train.shape[1]

BATCH_SIZE = 128  # definujeme velikost jedne davky dat
NUM_STEPS = 3000  # pocet treninkovych kroku
LR = 0.5  # learning rate

UNITS_HL = 1024  # pocet neuronu ve skryte vrstve
KEEP_PROB = 0.7  # pst zachovani signalu po dropoutu
# ###############################

graph = tf.Graph()
with graph.as_default():
    # drzak na data (sem budeme strkat jednotlive batche)
    X_train_tf = tf.placeholder(dtype=tf.float32,
                                shape=(BATCH_SIZE, num_features))
    y_train_tf = tf.placeholder(dtype=tf.float32,
                                shape=(BATCH_SIZE, num_labels))
    X_valid_tf = tf.constant(X_valid)
    X_test_tf = tf.constant(X_test)

    w1 = tf.Variable(tf.truncated_normal(shape=[num_features, UNITS_HL]))
    b1 = tf.Variable(tf.zeros(shape=[UNITS_HL]))
    w2 = tf.Variable(tf.truncated_normal(shape=[UNITS_HL, num_labels]))
    b2 = tf.Variable(tf.zeros(shape=[num_labels]))

    # novy parametr (abychom rozlisili kdy trenujeme a kdy skutecne predikujeme)
    def feed_forward(X, training=True):
        layer1 = tf.matmul(X, w1) + b1
        layer1 = tf.nn.relu(layer1)
        if training:  # pokud trenujeme
            # pridej dropout layer
            layer1 = tf.nn.dropout(x=layer1, keep_prob=KEEP_PROB)
        out_layer = tf.matmul(layer1, w2) + b2
        return out_layer

    logits = feed_forward(X_train_tf)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y_train_tf))

    # LR definovano nahore pro vetsi prehlednost
    optimizer = tf.train.GradientDescentOptimizer(LR).minimize(loss)

    train_pred = tf.nn.softmax(logits)
    valid_pred = tf.nn.softmax(feed_forward(X_valid_tf, training=False))  # netrenujeme
    test_pred = tf.nn.softmax(feed_forward(X_test_tf, training=False))  # netrenujeme


def accuracy(pred, labels):
    acc = (np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])
    return acc*100

def batch_generator(step, X, y):
    """Funkce pro tvorbu malych davek z velkeho datasetu"""
    # sahneme do datasetu a vytahneme data od tohoto bodu (offset)
    # az do body (offset + BATCH_SIZE)
    # Kdyz prelezeme pres velikost datasetu -> nevadi, zacneme od zacatku
    offset = (step * BATCH_SIZE) % (X.shape[0] - BATCH_SIZE)
    X_batch = X[offset:(offset + BATCH_SIZE)]
    y_batch = y[offset:(offset + BATCH_SIZE)]
    return X_batch, y_batch


with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized!")
    # NUM_STEPS definovano nohore pro vetsi prehlednost
    for step in range(NUM_STEPS):
        # vytvor davku dat
        X_batch, y_batch = batch_generator(step=step, X=X_train, y=y_train)
        # pekne ji zabal do slovniku
        feed_dict = {X_train_tf: X_batch,
                     y_train_tf: y_batch}
        _, l, pred = session.run([optimizer, loss, train_pred],
                                 feed_dict=feed_dict)  # nakrm davkou dat NN
        if step % 100 == 0:
            print("Loss at step {:d}: {:.4f}".format(step, l))
            print("Training accuracy: {:.1f} %".format(
                accuracy(pred, y_batch)))  # porovnamave pouze k davce labels
            print('Validation accuracy: {:.1f}%'.format(
                accuracy(valid_pred.eval(), y_valid)))
            print('-'*20)
    print('Test accuracy: {:.1f}%'.format(accuracy(test_pred.eval(), y_test)))
