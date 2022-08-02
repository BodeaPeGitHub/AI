import numbers
from random import shuffle

from PIL import ImageOps
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from sklearn import neural_network

import ANN
from main import split_into_test_and_train_data, evalMultiClass, plotConfusionMatrix

labels = ['original', 'sepia']
img_size = 64


def process_image(path):
    img = Image.open(path)
    img = np.asarray(img)
    img = list(map(lambda x: x[0], img))
    # img = ImageOps.grayscale(img)
    # img = img.resize(size=(img_size, img_size))
    return np.ravel(img) / 255.0


def get_data():
    input_data = []
    for i in range(1, 100):
        input_data.append([process_image('data/original/{}.jpg'.format(i)), labels.index("original")])
        input_data.append([process_image('data/sepia/{}-sepia.jpg'.format(i)), labels.index("sepia")])
    shuffle(input_data)
    output_data = [elem[1] for elem in input_data]
    input_data = list(map(lambda x: x[0], input_data))
    return input_data, output_data


def plot_data(outputs, outputs_names, title=""):
    plt.title(title)
    plt.hist(outputs, rwidth=0.8)
    plt.xticks(np.arange(len(outputs_names)), outputs_names)
    plt.show()


inputs, outputs = get_data()
train_inputs, train_outputs, validation_inputs, validation_outputs = split_into_test_and_train_data(inputs, outputs)
train_inputs = np.array(train_inputs)
train_outputs = np.array(train_outputs)
validation_inputs = np.array(validation_inputs)
validation_outputs = np.array(validation_outputs)
plot_data(train_outputs, labels)

classifier = ANN.NeuralNetwork(hidden_layer_sizes=(15, 20, 15), max_iter=5000)
# classifier = neural_network.MLPClassifier(hidden_layer_sizes=(10, 20, 15), max_iter=5000)
classifier.fit(train_inputs, train_outputs)
predictedLabels = classifier.predict(validation_inputs)
print(predictedLabels)
print(validation_outputs)
acc, prec, recall, cm = evalMultiClass(np.array(validation_outputs), predictedLabels, labels)
plotConfusionMatrix(cm, labels, "Sepia pictures classification.")
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', recall)
