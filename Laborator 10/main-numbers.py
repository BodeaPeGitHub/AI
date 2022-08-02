import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network

import ANN
from main import split_into_test_and_train_data, normalisation, evalMultiClass, plotConfusionMatrix


def loadDigitData():
    from sklearn.datasets import load_digits

    data = load_digits()
    inputs = data.images
    outputs = data['target']
    outputNames = data['target_names']

    # shuffle the original data
    noData = len(inputs)
    permutation = np.random.permutation(noData)
    inputs = inputs[permutation]
    outputs = outputs[permutation]

    return inputs, outputs, outputNames


def plot_data(outputs, outputs_names, title=""):
    plt.title(title)
    plt.hist(outputs, rwidth=0.8)
    plt.xticks(np.arange(len(outputs_names)), outputs_names)
    plt.show()


def flatten(mat):
    x = []
    for line in mat:
        for el in line:
            x.append(el)
    return x


input_data, output_data, output_names = loadDigitData()
train_input, train_output, validation_input, validation_output = split_into_test_and_train_data(input_data, output_data)
plot_data(train_output, output_names, title='Histogram of train samples.')
train_input_flatten = [flatten(elem) for elem in train_input]
validation_input_flatten = [flatten(elem) for elem in validation_input]
train_input_flatten_normalized, validation_input_flatten_normalized = normalisation(train_input_flatten,
                                                                                    validation_input_flatten)
classifier = neural_network.MLPClassifier()
classifier.fit(train_input_flatten_normalized, train_output)
predictedLabels = classifier.predict(validation_input_flatten_normalized)
acc, prec, recall, cm = evalMultiClass(np.array(validation_output), predictedLabels, output_names)

plotConfusionMatrix(cm, output_names, "Digit classification by tool.")
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', recall)

classifier = ANN.NeuralNetwork()
classifier.fit(train_input_flatten_normalized, train_output)
predictedLabels = classifier.predict(validation_input_flatten_normalized)
acc, prec, recall, cm = evalMultiClass(np.array(validation_output), predictedLabels, output_names)

plotConfusionMatrix(cm, output_names, "digit classification by me")
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', recall)