from random import random, shuffle
from math import exp, log
from statistics import mean
import numpy as np


class MyLogisticRegressionMultiLabel:

    def __init__(self):
        self.intercept = []
        self.coef_ = []

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def gradient_descendent(self, input_values, output_values, coef, label):
        new_coef = [0] * len(coef)
        for inp, out in zip(input_values, output_values):
            error = self.__sigmoid(self.__eval(inp, coef)) - 1 if out == label else self.__sigmoid(self.__eval(inp, coef))
            for i, xi in enumerate([1] + list(inp)):
                new_coef[i] += error * xi
        return new_coef

    def gradient_descendent_hinge_loss(self, input_values, output_values, coef, label):
        new_coef = [0] * len(coef)
        for inp, out in zip(input_values, output_values):
            z_computed = self.__eval(inp, coef)
            for i, xi in enumerate([1] + list(inp)):
                new_coef[i] += 0 if z_computed > 1 else (1 - z_computed) if z_computed > 0 else (0.5 - z_computed)
        return new_coef

    def fit(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.coef_ = []
        self.intercept = []
        labels = list(set(y))
        for label in labels:
            coef = [random() for _ in range(1 + len(x[0]))]
            for _ in range(no_epochs):
                change = self.gradient_descendent(x, y, coef, label)
                for i in range(len(coef)):
                    coef[i] -= learning_rate * change[i]
            self.intercept.append(coef[0])
            self.coef_.append(coef[1:])

    def fit_stocastic(self, x, y, learning_rate=0.001, no_epochs=1000):
        self.coef_ = []
        self.intercept = []
        indexes = [i for i in range(len(x))]
        labels = list(set(y))
        for label in labels:
            coef = [random() for _ in range(1 + len(x[0]))]
            for _ in range(no_epochs):
                shuffle(indexes)
                x = [x[i] for i in indexes]
                y = [y[i] for i in indexes]
                for inp, out in zip(x, y):
                    y_computed = self.__sigmoid(self.__eval(inp, coef))
                    error = y_computed - 1 if out == label else y_computed
                    for j in range(len(x[0])):
                        coef[j + 1] = coef[j + 1] - learning_rate * error * inp[j]
                    coef[0] = coef[0] - learning_rate * error
            self.intercept.append(coef[0])
            self.coef_.append(coef[1:])

    def fit_stocastic_with_hinge_loss(self, x, y, learning_rate=0.001, no_epochs=1000):
        pass

    def __eval(self, inp, coef):
        return sum([wi * xi for wi, xi in zip(coef[1:], inp)]) + coef[0]

    def predict_one_sample(self, sample_features):
        predictions = []
        for label, coef in enumerate(self.coef_):
            computed_value = self.__eval(sample_features, [self.intercept[label]] + coef)
            predictions.append(self.__sigmoid(computed_value))
        return predictions.index(max(predictions))

    def predict(self, inputs):
        return [self.predict_one_sample(sample) for sample in inputs]

    def score(self, inputs, outputs):
        corect = 0
        for predicted, real in zip(self.predict(inputs), outputs):
            if predicted == real:
                corect += 1
        return float(corect) / len(outputs)
