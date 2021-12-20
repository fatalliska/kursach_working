import numpy as np
import os
import tensorflow.keras as keras
from keras import layers as L
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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
      matrixes.append(df.iloc[:, 250:627].values)
y = np.array([True for i in range(len(matrixes))])
length = len(matrixes)

folder = []
pas = "/home/alisa/withoutErrors/"
x = np.arange(0, 15.08, 0.04)
for i in os.walk(pas):
    folder.append(i)
for i in folder[0][2]:
    df = pd.read_csv(pas + i, header=None)
    matrixes.append(df.iloc[:, 250:627].values)
y = np.append(y, [False for i in range(len(matrixes)-length)])
results = np.array([np.mean(matrixes[i], axis=1) for i in range(len(matrixes))])
print(results.shape)
x_train, x_test, y_train, y_test = train_test_split(results, y, test_size = 0.1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# class TimeFrameConv(keras.layers.Layer):
#     def __init__(self, units=32, input_dim=32):
#         super(TimeFrameConv, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )
#
#     @tf.autograph.experimental.do_not_convert
#     def call(self, inputs, training):
#         #result = np.zeros(32)
#         result = []
#         tf.print(inputs)
#         for i in range(inputs[0].shape[0]):
#                frame = inputs[0][i]
#                result.append(tf.nn.conv1d(input=frame, filters=1, stride=1, padding='same'))
#                #result[i] = np.mean(np.array([sum(frame[j:j+30])/29 for j in range(0, 29, frame.shape[0])]))
#         print(result)
#
init = 'uniform'
act = 'softmax'
opt = keras.optimizers.Adam(learning_rate=1)
input = L.Input(shape=(32))  # задаем вход
#x = L.BatchNormalization()(input_tensor) # применение нейрона к входу
#x = L.Conv2D()
#x = TimeFrameConv()(input_tensor)
x = L.Dense(32, kernel_initializer=init, activation=act)(input)
x = L.Dense(16, kernel_initializer=init, activation=act)(x)
output = L.Dense(2, kernel_initializer=init, activation='sigmoid')(input)
model = keras.Model(input, output)
model.compile(optimizer=opt,loss='binary_crossentropy',  metrics=["categorical_accuracy"])
history1 = model.fit(x_train, y_train,batch_size=200,epochs=3,validation_data = (x_test,y_test))