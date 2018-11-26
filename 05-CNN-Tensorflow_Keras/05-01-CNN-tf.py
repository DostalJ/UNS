import tensorflow as tf  # tensorflow pro ML
from pickle import load  # pickle load pro nacteni dat

# toto je jen skareda droobnost, ktera nas zbavi otravne varujici hlasky
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PATH = './data/MNIST.pkl'  # cesta k datum
# otevreme soubor a pojmenujeme ho 'pkl_file'
with open(file=PATH, mode='rb') as pkl_file:
    save = load(file=pkl_file)  # vytahneme ze souboru velky slovnik s daty
    X_train = save['X_train']
    y_train = save['y_train']
    X_valid = save['X_valid']
    y_valid = save['y_valid']
    X_test = save['X_test']
    y_test = save['y_test']
    del save  # uvolnime pamet

print('#'*40)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_valid:", X_valid.shape)
print("Shape of y_valid:", y_valid.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print('#'*40)

# ##########################################
IMAGE_SIZE = X_train.shape[1]  # velikost obrazku
NUM_CHANNELS = X_train.shape[3]  # pocet barevnych kanalu
NUM_CLASSES = y_train.shape[1]  # pocet trid pro klasifikaci

NUM_STEPS = 301  # pocet kroku optimalizace
BATCH_SIZE = 128  # pocet obrazku na ktere se koukneme pri jednom kroku

LR = 0.001  # learning rate

PATCH_SIZE = 5  # velikost jednoho filteru
CH = 32  # pocet vrstev po prvni CNN transformaci
FCL = 1024  # pocet skrytych vrstev
# ##########################################


def conv_layer(input_, patch_size, channels_in, channels_out):
    """
    Convolution layer.
    Transformes using convolutional filter + adds bias + activates using ReLu.
    Arguments:
        input_: input tensor
        patch_size: filter size
        channels_in: number of input channels (feature of 'input_')
        channels_out: number of output channels
    """
    # definujeme vahu filteru
    w = tf.Variable(tf.truncated_normal(  # nahodny vyber z normalniho rozdeleni
        shape=[patch_size, patch_size, channels_in, channels_out],
        stddev=0.1))  # s timto rozptylem
    b = tf.Variable(tf.constant(value=0.1,  # bias inicializovan s hodnotou 0.1
                                shape=[channels_out]))  # dlouhy 'channels_out
                                # ma to jeden rozmer, protoze si s tim umi TF
                                # pohrat a pricist spravne
    conv = tf.nn.conv2d(input=input_,  # co bude transformovano
                        filter=w,  # jaky uzijeme filter
                        strides=[1, 1, 1, 1],  # velikost kroku ve 4D
                        padding='SAME',  # jak osetrime okraje
                        # "same" results in padding the input such that the
                        # output has the same length as the original input.
                        data_format="NHWC")  # batch_size,height,width,channels
    act = tf.nn.relu(conv + b)  # aktivujeme pomoci ReLu
    return act  # vratime aktivovaly tensor


def fc_layer(input_, channels_in, channels_out, relu=True):
    """
    Fully-connected layer.
    Multiplies by weights + adds bias + (optionaly) activates using ReLu.
    Arguments:
        input_: input tensor
        channels_in: number of input channels (feature of 'input_')
        channels_out: number of output channels
    """
    # definuje vahy o rozmerech ['channels_in', 'channels_out']
    w = tf.Variable(tf.truncated_normal(shape=[channels_in, channels_out],
                                        stddev=0.1))  # rozptyl
    b = tf.Variable(tf.constant(value=0.1,  # bias inicializovan s hodnotou 0.1
                                shape=[channels_out]))  # dlouhy 'channels_out
    layer = tf.matmul(input_, w) + b  # vynasobime a pricteme bias
    if relu:  # pokud chceme pouzit ReLu aktivaci
        layer = tf.nn.relu(layer)  # zaktivujeme
    return layer  # vratime vysledny tensor


def print_info(step=None, loss=None, train_acc=None, valid_acc=None, test_acc=None):
    if step is not None:  # pokud dodam 'step' (muze byt step=0, proto jina notace)
        print('Step {:d}:'.format(step))  # vypisu
    if loss:  # pokud dodam 'loss'
        print('CrossEntropy: {:.3f}'.format(loss))  # vypisu
    if train_acc:  # pokud dodam 'train_acc'
        print('Training accuracy: {:.1f} %'.format(train_acc*100))  # vypisu
    if valid_acc:  # pokud dodam 'valid_acc'
        print('Validation accuracy: {:.1f} %'.format(valid_acc*100))  # vypisu
    if test_acc:  # pokud dodam 'test_acc'
        print('Test accuracy: {:.1f} %'.format(test_acc*100))  # vypisu
    print('-'*30)


def feed_forward(X):
    """Network architecture"""
    # uzijeme konvolucni vrstvu
    conv = conv_layer(input_=X,
                      patch_size=PATCH_SIZE,
                      channels_in=NUM_CHANNELS,
                      channels_out=CH)

    # nyni z 4D tensoru prevedeme na 2D, abychom mohli pouzit FCL
    # z [128, 7, 7, 16] na [128, 784] = [128, 7*7*16]
    shape = conv.get_shape().as_list()  # zjistime rozmery tensoru
    flattened = tf.reshape(tensor=conv,  # zplacatime tento tensor
                           shape=[-1, shape[1]*shape[2]*shape[3]])

    # uzijeme plne propojenou vrstvu
    fc1 = fc_layer(input_=flattened,
                   channels_in=shape[1]*shape[2]*shape[3],
                   channels_out=FCL)  # hloubka po pronasobeni
    # uzijeme plne propojenou vrstvu
    logits = fc_layer(input_=fc1,
                      channels_in=FCL,
                      channels_out=NUM_CLASSES,  # vysledna hloubka tensoru
                      relu=False)  # nicim neaktivujeme
    return logits  # vratime neaktivovanou posledni vrstvu


def batch_generator(step, X, y):
    """Batch generator"""
    # sahneme do datasetu a vytahneme data od tohoto bodu (offset)
    # az do body (offset + BATCH_SIZE)
    # Kdyz prelezeme pres velikost datasetu -> nevadi, zacneme od zacatku
    offset = (step * BATCH_SIZE) % (X.shape[0] - BATCH_SIZE)
    X_batch = X[offset:(offset + BATCH_SIZE)]
    y_batch = y[offset:(offset + BATCH_SIZE)]
    return X_batch, y_batch


def model():
    tf.reset_default_graph()  # jinak bych zapisoval summaries do stareho grafu
    sess = tf.Session()  # TensorFlow session

    # drzak pro data
    X_tf = tf.placeholder(shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS],
                          dtype=tf.float32)
    y_tf = tf.placeholder(shape=[None, NUM_CLASSES],
                          dtype=tf.float32)

    logits = feed_forward(X=X_tf)  # necham projit data celym grafem

    # spocitam 'loss'
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits,
        labels=y_tf))

    # udelam jeden treninkovy krok, tj. zoptimalizuji
    train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)

    prediction = tf.nn.softmax(logits)  # zaktivuji --> pravdepodobnosti trid

    # kde vsude jsem se trefil s predikci
    correct_prediction = tf.equal(tf.argmax(prediction, axis=1),
                                  # tf.argmax prevede na index, tj. tridu
                                  tf.argmax(y_tf, axis=1))
    # prevedu na jiny datovy typ
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # inicializuji promenne (TF stuff)
    sess.run(tf.global_variables_initializer())

    print('Training...\n' + '-'*30)
    for step in range(NUM_STEPS):  # iteruji pres pocet kroku
        batch = batch_generator(step=step, X=X_train, y=y_train)  # davka dat
        # davka dat v peknem formatu
        feed_dict = {X_tf: batch[0], y_tf: batch[1]}
        sess.run(train_step, feed_dict=feed_dict)  # zoptimalizuji

        if step % 50 == 0:  # pri kazdem 50. kroku
            # spocitam training accuracy
            [train_acc, loss] = sess.run([accuracy, cross_entropy],
                                         feed_dict=feed_dict)
            # spocitam valid accuracy
            valid_acc = sess.run(accuracy,
                                 feed_dict={X_tf: X_valid, y_tf: y_valid})
            # a vypisi na obrazovku
            print_info(step=step,
                       loss=loss,
                       train_acc=train_acc,
                       valid_acc=valid_acc)
    # nakonec spocitam jak bych si vedl na testovaci sade
    test_acc = sess.run(accuracy,
                        feed_dict={X_tf: X_test, y_tf: y_test})
    print_info(test_acc=test_acc)  # a vypisu


model()  # spustim model
