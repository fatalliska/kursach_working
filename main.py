import numpy as np
import os
import tensorflow.keras as keras
from keras import layers as L
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa

#получаем список всех папок и файлов в них
folder = []
pas = "/home/alisa/withErrors/"
x = np.arange(0, 15.08, 0.04)
for i in os.walk(pas):
    folder.append(i)
matrixes = []
for i in folder[0][2]:
    if not 'time' in i:
      df = pd.read_csv(pas + i, header=None)
      matrixes.append(df.iloc[:, 501:626].values)
y = np.array([True for i in range(len(matrixes))])
length = len(matrixes)

folder = []
pas = "/home/alisa/withoutErrors/"
x = np.arange(0, 15.08, 0.04)
for i in os.walk(pas):
    folder.append(i)
for i in folder[0][2]:
    df = pd.read_csv(pas + i, header=None)
    matrixes.append(df.iloc[:, 501:626].values)
y = np.append(y, [False for i in range(len(matrixes)-length)])
results = np.array([np.mean(matrix, axis=0) for matrix in matrixes])
x_train, x_test, y_train, y_test = train_test_split(results, y, test_size = 0.1)
if False in y_test:
    print('ok')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

init = keras.initializers.RandomUniform()
act = 'relu'
opt = keras.optimizers.Adam(learning_rate=1)
input = L.Input(shape=(results.shape[1]))
x = L.Dense(8, kernel_initializer=init, activation=act)(input)
output = L.Dense(2, kernel_initializer=init, activation='sigmoid')(x)
model = keras.Model(input, output)
model.compile(optimizer=opt,loss='binary_crossentropy',  metrics=["categorical_accuracy", keras.metrics.Recall(),
                                                                  tfa.metrics.F1Score(num_classes=2, average='weighted'),
                                                                  keras.metrics.TruePositives(), keras.metrics.TrueNegatives(),
                                                                  keras.metrics.FalsePositives(), keras.metrics.FalseNegatives()])
history1 = model.fit(x_train, y_train, batch_size=200, epochs=1, validation_data=(x_test, y_test))
