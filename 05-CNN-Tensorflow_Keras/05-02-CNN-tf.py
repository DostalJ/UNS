from os import environ
from pickle import load

import tensorflow as tf

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = './data/MNIST.pkl'
with open(file=PATH, mode='rb') as pkl_file:
    save = load(file=pkl_file)
    X_train = save['X_train']
    y_train = save['y_train']
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    X_test = save['X_test']
    y_test = save['y_test']
    del save

print('#' * 40)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_valid:", X_valid.shape)
print("Shape of y_valid:", y_valid.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print('#' * 40)

# ##########################################
IMAGE_SIZE = X_train.shape[1]
NUM_CHANNELS = X_train.shape[3]
NUM_CLASSES = y_train.shape[1]

NUM_STEPS = 401
BATCH_SIZE = 128

LR = 0.001
KEEP_PROB = 0.6

FILTER_1 = 3  # velikost prvniho filteru
CH_1 = 32  # pocet vrstev po prvni CNN transformaci
FILTER_2 = 5  # velikost druheho filteru
CH_2 = 64  # pocet vrstev po druhe CNN transformaci
FCL_1 = 1024  # pocet skrytych vrstev po FCL


# ##########################################


def conv_layer(input_, patch_size, channels_in, channels_out, name='conv'):
    """
    Convolution layer.
    Transformes using convolutional filter + adds bias + activates using ReLu.
    Arguments:
        input_: input tensor
        patch_size: filter size
        channels_in: number of input channels (feature of 'input_')
        channels_out: number of output channels
        name: name of the operation for TensorBoard
    """
    with tf.name_scope(name):  # definujeme jako jednu operaci
        w = tf.Variable(tf.truncated_normal(
            shape=[patch_size, patch_size, channels_in, channels_out],
            stddev=0.1),
            name='W')  # jmeno promenne
        b = tf.Variable(tf.constant(value=0.1,
                                    shape=[channels_out]),
                        name='b')  # jmeno promenne
        conv = tf.nn.conv2d(input=input_,
                            filter=w,
                            strides=[1, 1, 1, 1],
                            padding='SAME',
                            data_format="NHWC")
        act = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", w)  # pridame do summary
        tf.summary.histogram("biases", b)  # pridame do summary
        tf.summary.histogram("activations", act)  # pridame do summary
        return act


def fc_layer(input_, channels_in, channels_out, relu=True, name='fc'):
    """
    Fully-connected layer.
    Multiplies by weights + adds bias + (optionaly) activates using ReLu.
    Arguments:
        input_: input tensor
        channels_in: number of input channels (feature of 'input_')
        channels_out: number of output channels
        name: name of the operation for TensorBoard
    """
    with tf.name_scope(name):  # definujeme jako jednu operaci
        w = tf.Variable(tf.truncated_normal(shape=[channels_in, channels_out],
                                            stddev=0.1),
                        name='W')  # jmeno promenne
        b = tf.Variable(tf.constant(value=0.1,
                                    shape=[channels_out]),
                        name='b')  # jmeno promenne
        layer = tf.matmul(input_, w) + b
        if relu:
            layer = tf.nn.relu(layer)

        tf.summary.histogram("weights", w)  # pridame do summary
        tf.summary.histogram("biases", b)  # pridame do summary
        tf.summary.histogram("activations", layer)  # pridame do summary
        return layer


def print_info(step=None, loss=None, train_acc=None, valid_acc=None, test_acc=None):
    if step is not None:
        print('Step: {:03d}'.format(step))
    if loss:
        print('Cross Entropy: {:.3f}'.format(loss))
    if train_acc:
        print('Training accuracy: {:.1f} %'.format(train_acc * 100))
    if valid_acc:
        print('Validation accuracy: {:.1f} %'.format(valid_acc * 100))
    if test_acc:
        print('Test accuracy: {:.1f} %'.format(test_acc * 100))
    print('-' * 30)


def feed_forward(X, keep_prob):
    conv1 = conv_layer(input_=X,
                       patch_size=FILTER_1,
                       channels_in=NUM_CHANNELS,
                       channels_out=CH_1,
                       name='conv1')
    conv1 = tf.nn.max_pool(value=conv1,
                           ksize=[1, 2, 2, 1],  # velikost max_pool filtru
                           strides=[1, 2, 2, 1],  # posouvame se ob jedno
                           padding='SAME',
                           data_format="NHWC",
                           name='max_pool1')
    conv2 = conv_layer(input_=conv1,
                       patch_size=FILTER_2,
                       channels_in=CH_1,
                       channels_out=CH_2,
                       name='conv2')
    conv2 = tf.nn.max_pool(value=conv2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           data_format="NHWC",
                           name='max_pool2')

    # from [16, 7, 7, 16] to [16, 784] = [16, 7*7*16]
    shape = conv2.get_shape().as_list()
    flattened = tf.reshape(tensor=conv2,
                           shape=[-1, shape[1] * shape[2] * shape[3]])

    # pridam dropout layer
    flattened = tf.nn.dropout(x=flattened,
                              keep_prob=keep_prob,
                              name='dropout')

    fc1 = fc_layer(input_=flattened,
                   channels_in=shape[1] * shape[2] * shape[3],
                   channels_out=FCL_1,
                   relu=True,
                   name='fc1')
    logits = fc_layer(input_=fc1,
                      channels_in=FCL_1,
                      channels_out=NUM_CLASSES,
                      relu=False,
                      name='fc2')
    return logits


def batch_generator(step, X, y):
    offset = (step * BATCH_SIZE) % (X.shape[0] - BATCH_SIZE)
    X_batch = X[offset:(offset + BATCH_SIZE)]
    y_batch = y[offset:(offset + BATCH_SIZE)]
    return X_batch, y_batch


def model(log_name):
    tf.reset_default_graph()
    sess = tf.Session()

    keep_prob = tf.placeholder(dtype=tf.float32)

    X_tf = tf.placeholder(shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                          dtype=tf.float32,
                          name='X')
    y_tf = tf.placeholder(shape=[None, NUM_CLASSES],
                          dtype=tf.float32,
                          name='y')

    tf.summary.image(tensor=X_tf,
                     max_outputs=3,
                     name='input_image')

    logits = feed_forward(X=X_tf, keep_prob=keep_prob)

    with tf.name_scope('cross_entropy'):  # definujeme jako jednu operaci
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=y_tf))
        # pridame do summary
        tf.summary.scalar(tensor=cross_entropy, name='cross_entropy')

    with tf.name_scope('train'):  # definujeme jako jednu operaci
        train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

    with tf.name_scope('predict'):  # definujeme jako jednu operaci
        prediction = tf.nn.softmax(logits)

    with tf.name_scope('accuracy'):  # definujeme jako jednu operaci
        correct_prediction = tf.equal(tf.argmax(prediction, axis=1),
                                      tf.argmax(y_tf, axis=1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # pridame do summary
        tf.summary.scalar(tensor=accuracy, name='accuracy')

    sess.run(tf.global_variables_initializer())

    summ = tf.summary.merge_all()  # spojime vsechny summary

    # definuje dve zvlastni summary pro trenink a validaci zvlast
    # (pokud chci ukladat i vysledkyvalidace, inteligentneji to nejde)
    train_writer = tf.summary.FileWriter('./logs/{}/train'.format(log_name))
    valid_writer = tf.summary.FileWriter('./logs/{}/valid'.format(log_name))
    train_writer.add_graph(sess.graph)  # pridame strukturu grafu

    print('Training...\n' + '-' * 30)
    for step in range(NUM_STEPS):
        batch = batch_generator(step=step, X=X_train, y=y_train)
        # pri treninku uzivam KEEP_PROB < 1.0
        batch_feed_dict = {X_tf: batch[0], y_tf: batch[1], keep_prob: KEEP_PROB}
        sess.run(train_step, feed_dict=batch_feed_dict)

        if step % 50 == 0:
            [train_acc, loss] = sess.run([accuracy, cross_entropy],
                                         feed_dict=batch_feed_dict)
            # pri validaci KEEP_PROB = 1.0
            [valid_acc, s] = sess.run([accuracy, summ],
                                      feed_dict={X_tf: X_valid,
                                                 y_tf: y_valid,
                                                 keep_prob: 1.0})
            print_info(step=step,
                       loss=loss,
                       train_acc=train_acc,
                       valid_acc=valid_acc)
            valid_writer.add_summary(s, step)  # zapisu validacni summary
        if step % 5 == 0:  # pri kazdem 5. kroku aktualizuji treninkove summary
            [train_acc, s] = sess.run([accuracy, summ],
                                      feed_dict=batch_feed_dict)
            train_writer.add_summary(s, step)

    # testuji opet s KEEP_PROB = 1.0
    test_acc = sess.run(accuracy,
                        feed_dict={X_tf: X_test,
                                   y_tf: y_test,
                                   keep_prob: 1.0})
    print_info(test_acc=test_acc)


model('02-Adam')
