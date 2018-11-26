from pickle import load

from keras.callbacks import TensorBoard
from keras.layers import (Conv2D, Activation, Flatten,
                          Dense)
from keras.models import Sequential


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

print('#' * 40)
print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of X_valid:", X_valid.shape)
print("Shape of y_valid:", y_valid.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_test:", y_test.shape)
print('#' * 40)

# ##########################################
IMAGE_SIZE = X_train.shape[1]  # velikost obrazku
NUM_CHANNELS = X_train.shape[3]  # pocet barevnych kanalu
NUM_CLASSES = y_train.shape[1]  # pocet trid pro klasifikaci

IMAGE_SIZE_TUPLE = (IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)  # rozmery obrazku

BATCH_SIZE = 64
NUM_EPOCHS = 3  # (NUM_STEPS * BATCH_SIZE) / X_train.shape[0]

LR = 0.001  # learning rate
# ##########################################


model = Sequential()  # definujeme vypocetni graf
model.add(Conv2D(filters=21,  # pocet vystupnich vrstev
                 kernel_size=(5, 5),  # velikost prvniho filteru
                 input_shape=IMAGE_SIZE_TUPLE))  # rozmery vstupu
model.add(Activation('relu'))  # zaktivujeme pomoci 'relu'
model.add(Flatten())  # prevedeme do 1D vektoru
model.add(Dense(units=1024))  # aplikujeme FCL
model.add(Activation('relu'))
model.add(Dense(units=NUM_CLASSES))  # konecna vystupni vrstva
model.add(Activation('softmax'))  # softmax aktivace

model.compile(optimizer='adam',  # dejinujeme optimizer
              loss='categorical_crossentropy',  # ztratovou fce
              metrics=['accuracy'])  # pripadne metriku presnosti

# pokud chceme zapisovat zajimave info o treninku, definujeme TB objekt
myTB = TensorBoard(log_dir='./logs',  # sem se budou zapisovat data
                   # zapise se i struktura grafu (muze byt velky a zpomaly
                   # trenink)
                   write_graph=True,
                   histogram_freq=1)  # kazdou epochu zapise loss a acc

print(model.summary())  # vypise sumarizaci nasi definice

# natrenuje model na X_trai, y_train
model.fit(x=X_train, y=y_train,
          batch_size=BATCH_SIZE,  # velikost davek adt
          epochs=NUM_EPOCHS,  # pocet pohledu na cely dataset
          verbose=1,  # pekny vypis v prubehu treninku
          callbacks=[myTB],  # pomoci tohoto objektu bude zapisovat
          validation_data=(X_valid, y_valid))  # validacni data (tuple)

l, acc = model.evaluate(x=X_test, y=y_test)  # loss a acc pro testovaci data
print('\nTest accuracy: {:.2f} %'.format(acc * 100))  # vypise info o testu
