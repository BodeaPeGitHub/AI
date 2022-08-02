import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import keras
from PIL import ImageOps
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np

labels = ['original', 'sepia']
img_size = 64


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)


all_data = get_data('data')
indexes = [i for i in range(len(all_data))]
train_sample = np.random.choice(indexes, int(0.8 * len(all_data)), replace=False)
validation_sample = [i for i in indexes if i not in train_sample]
train_data = [all_data[i] for i in train_sample]
validation_data = [all_data[i] for i in validation_sample]

l = []
for i in train_data:
    l.append('Original.' if i[1] == 0 else 'Sepia.')

sns.set_style('darkgrid')
sns.countplot(x=l)

plt.figure(figsize=(5, 5))
plt.imshow(train_data[0][0])
plt.title(labels[train_data[0][1]])
plt.show()

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train_data:
    x_train.append(feature)
    y_train.append(label)

for feature, label in validation_data:
    x_val.append(feature)
    y_val.append(label)

# Normalize the data.
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=30,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32, 3, padding="same", activation="relu", input_shape=(img_size, img_size, 3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

from keras.optimizers import adam_v2

opt = adam_v2.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50)

acc = history.history['accuracy']
print(acc)
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

