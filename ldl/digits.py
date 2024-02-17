# This is the third example program from the 'Introduction
# to "Learning Deep Learning"' training course at:
# https://www.nvidia.com/en-us/on-demand/session/gtcspring23-dlit52044/?playlistId=playList-5906792e-0f3d-436b-bc30-3abf911a95a6
# by Magnus Ekman.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# standardize the images
mean = np.mean(train_images)
stddev = np.std(train_images)
train_images = (train_images - mean) / stddev
test_images = (test_images - mean) / stddev

# one-hot encode labels
one_hot_train_labels = to_categorical(train_labels, num_classes=10)
one_hot_test_labels = to_categorical(test_labels, num_classes=10)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(25, activation='tanh'),
    keras.layers.Dense(10, activation='sigmoid')])

opt = keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer = opt, loss='mean_squared_error', metrics =['accuracy'])

history = model.fit(train_images, one_hot_train_labels,
                    validation_data=(test_images, one_hot_test_labels),
                    epochs=10, batch_size=64, verbose=2, shuffle=True)

TEST_NUM = 0
plt.imshow(test_images[TEST_NUM], cmap=plt.get_cmap('gray'))
plt.show()
print('Ground truth:', test_labels[TEST_NUM])
prediction = model.predict(test_images[TEST_NUM:(TEST_NUM + 1)])
print('Prediction:', prediction[0].argmax())
print(prediction)


