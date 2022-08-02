import random


class MySGDRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    # simple stochastic GD
    def fit(self, x, y, learningRate=0.001, noEpochs=1000):
        # self.coef_ = [0.0 for _ in range(len(x[0]) + 1)]    #beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]  # beta or w coefficients
        for epoch in range(noEpochs):
            # TBA: shuffle the trainind examples in order to prevent cycles
            temp_coeficients = [] + self.coef_
            for i in range(len(x)):  # for each sample from the training data
                ycomputed = self.eval(x[i])  # estimate the output
                crtError = ycomputed - y[i]  # compute the error for the current sample
                for j in range(0, len(x[0])):  # update the coefficients
                    self.coef_[j] = self.coef_[j] - learningRate * crtError * x[i][j]
                self.coef_[len(x[0])] = self.coef_[len(x[0])] - learningRate * crtError * 1

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def fit_with_batch(self, x, y, learningRate=0.001, noEpochs=10000):
        self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]
        for epoch in range(noEpochs):
            temp_coeficients = [0] * (len(x[0]) + 1)
            for i in range(len(x)):
                ycomputed = self.eval(x[i])
                crtError = ycomputed - y[i]
                for j in range(0, len(x[0])):
                    temp_coeficients[j] = temp_coeficients[j] - learningRate * crtError * x[i][j]
                temp_coeficients[len(x[0])] = temp_coeficients[len(x[0])] - learningRate * crtError * 1
            for index in range(len(self.coef_)):
                self.coef_[index] += temp_coeficients[index]

        self.intercept_ = self.coef_[-1]
        self.coef_ = self.coef_[:-1]

    def eval(self, xi):
        yi = self.coef_[-1]
        for j in range(len(xi)):
            yi += self.coef_[j] * xi[j]
        return yi

    def predict(self, x):
        return [self.eval(xi) for xi in x]
