import pickle  # pickle pro nacteni dat
import numpy as np  # numpy pro praci s maticemi a vektory
import tensorflow as tf  # tensorflow pro ML

PATH = './data/MNIST.pkl'  # cesta k datum
# otevreme soubor a pojmenujeme ho 'pkl_file'
with open(file=PATH, mode='rb') as pkl_file:
    # vytahneme ze souboru velky slovnik s daty
    save = pickle.load(file=pkl_file)
    X_train = save['X_train']  # ze slovniku vezmene treninkove features
    y_train = save['y_train']  # a taky labels
    del save  # vymazeme slovnik (je uz k nicemu a velky)

print('#'*40)  # pro hezci vypis
print("Shape of X_train:", X_train.shape)  # rozmery dat
print("Shape of y_train:", y_train.shape)  # rozmery labels
print('#'*40)  # pro hezci vypis

num_features = X_train.shape[1]  # velikost feature vektoru
num_labels = y_train.shape[1]  # pocet skupin do kterych klasifikujeme

graph = tf.Graph()  # definujeme vypocetni graf
with graph.as_default():
    X_train_tf = tf.constant(X_train)  # nacteme data do TF vypocetniho grafu
    y_train_tf = tf.constant(y_train)  # nacteme data do TF vypocetniho grafu

    # vahy: je to 'tf.Variable' inicializovana normalnim rozdelenim
    weights = tf.Variable(tf.truncated_normal(shape=[num_features, num_labels]))
    # biases: je to 'tf.Variable' inicializovana vsude s nulami
    biases = tf.Variable(tf.zeros(shape=[num_labels]))

    logits = tf.matmul(X_train_tf, weights) + biases  # neaktivovany vystup z grafu

    # neaktivujeme ho, protoze ted pro vypocet 'loss' pouzivame funkci, ktera
    # si vystup zaktivuje sama --> umi to mnohem efektivneji!
    # TF ani neposkytuje moznost nejprve rucne aktivovat a pak vypocitat 'loss'
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=y_train_tf))

    # optimalizujeme pomoci GradientDescent s LR = 0.5
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # zaktivujeme 'logits' pomoci softmax fce --> predikce
    train_pred = tf.nn.softmax(logits)

# definujeme accuracy (metrika uspesnosti pro klasifikaci)
# acc == 1 --> vse jsme trefili
# acc == 0 --> nic jsme netrefili 
def accuracy(pred, labels):
    acc = (np.sum(np.argmax(pred, 1) == np.argmax(labels, 1)) / pred.shape[0])
    return acc*100


# zacneme neco pocitat na grafu definovanem vyse
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()  # inicializujeme vsechny promenne
    print("Initialized!")  # napiseme, ze jsme je inicializovali :)
    for step in range(250):  # opakujeme 250 krat
        # na grafu vypocitame 'optimizer', 'loss', 'train_pred'
        # 'optimizer' --> prenastavi vahy a biases
        # 'loss' --> (Vypocita se uz pri volani 'optimizer'. My ji ale chceme
        #             vypsat na obrazovku, proto ji potrebujeme vyextrahovat z
        #             grafu)
        # 'train_pred' --> chceme vypsat, proto museme vyndat z grafu
        _, l, pred = session.run([optimizer, loss, train_pred])
        if step % 25 == 0:  # kazdy 25-ty krok vypiseme uzitecne informace
            print("Loss at step {:d}: {:.4f}".format(step, l))  # loss
            print("Training accuracy: {:.1f} %".format(  # presnost
                accuracy(pred, y_train)))
            print('-'*20)  # pro hezci vypis
