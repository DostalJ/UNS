from pickle import load

from tensorflow.contrib.keras.api.keras.callbacks import TensorBoard
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.api.keras.models import Sequential

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

# ###############################
num_features = X_train.shape[1]
num_labels = y_train.shape[1]

BATCH_SIZE = 512
NUM_EPOCHS = 10  # (NUM_STEPS * BATCH_SIZE) / X_train.shape[0]

UNITS_HL = 1024
KEEP_PROB = 0.7
# ###############################

model = Sequential()  # definujeme model
# pridame prvni FC vrstvu
# u prvni musime specifikovat vstup, u ostatnich nemusime
model.add(Dense(input_dim=num_features,
                units=UNITS_HL,  # pocet neuronu
                activation='relu'))  # aktivacni fce
# dropout vrstva (narozdil od cisteho TF je jejim argumentem proporce neuronu,
# ktere se nahodne smazou)
model.add(Dropout(rate=(1-KEEP_PROB)))
# vystupni vrstva zakoncena softmax aktivaci
model.add(Dense(units=num_labels, activation='softmax'))

# model tzv. 'zkompilujeme
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
print('\nTest accuracy: {:.2f} %'.format(acc*100))  # vypise info o testu
