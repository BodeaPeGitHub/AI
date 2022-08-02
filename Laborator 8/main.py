# %%

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from MyRegressor import MySGDRegression
from sklearn.preprocessing import StandardScaler

plt.style.use({'figure.facecolor': 'white'})


# %%
def load_data(filename, input_variable_name, output_variable_name):
    file = pd.read_csv(filename)
    inputs = []
    for input_variable_name in input_variable_name:
        inputs.append([float(val) for val in file[input_variable_name]])
    output = [float(val) for val in file[output_variable_name]]
    return inputs, output


input_data, output_data = load_data('data/v1_world-happiness-report-2017.csv', ['Economy..GDP.per.Capita.', 'Freedom'],
                                    'Happiness.Score')


# %%

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


plotDataHistogram(input_data[0], 'Economy GDP per Capital.')
plotDataHistogram(input_data[1], 'Freedom')
plotDataHistogram(output_data, 'Happiness')


# %%

def plot_for_linearity(inputs, outputs, input_label, output_label, title=""):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.title(title)
    plt.show()


plot_for_linearity(input_data[0], output_data, "GDA capital", "Happiness", 'GDA vs Happiness')


# %%

def split_in_train_and_test_samples(inputs, outputs):
    np.random.seed(5)
    train_sample_indexes = np.random.choice([i for i in range(len(outputs))], int(0.8 * len(outputs)), replace=False)
    validation_sample_indexes = [i for i in range(len(outputs)) if i not in train_sample_indexes]
    train_inputs = []
    validation_inputs = []
    for inp in inputs:
        train_inputs.append([inp[i] for i in train_sample_indexes])
        validation_inputs.append([inp[i] for i in validation_sample_indexes])
    train_outputs = [outputs[i] for i in train_sample_indexes]
    validation_outputs = [outputs[i] for i in validation_sample_indexes]
    return train_inputs, train_outputs, validation_inputs, validation_outputs


train_inputs, train_outputs, validation_inputs, validation_outputs = split_in_train_and_test_samples(input_data,
                                                                                                     output_data)
plt.plot(train_inputs[0], train_outputs, 'ro', label='Train')
plt.plot(validation_inputs[0], validation_outputs, 'g*', label='Test')
plt.xlabel("GDA per capital")
plt.ylabel("Happiness")
plt.title('Train and test data.')
plt.show()


# %% md
# Learning by tool for single feature.

# %%

def learn_by_tool_single_feature():
    xx = [[feature] for feature in train_inputs[0]]
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor.fit(xx, train_outputs)
    return regressor.intercept_[0], regressor.coef_[0]


w0, w1 = learn_by_tool_single_feature()
print('Lear model by tool returned: f(x) = ', w0, ' + ', w1, ' * x')


# %% md
# Learning by my code for single feature.

# %%

def learn_by_my_code_single_feature():
    xx = [[feature] for feature in train_inputs[0]]
    regressor = MySGDRegression()
    regressor.fit_with_batch(xx, train_outputs)
    error = 0.0
    for real_output, computed_output in zip(validation_outputs, regressor.predict([e] for e in validation_inputs[0])):
        error += (real_output - computed_output) ** 2
    error /= len(validation_outputs)
    return regressor.intercept_, regressor.coef_[0], error


w0, w1, error_by_me = learn_by_my_code_single_feature()
print('Lear model by me returned: f(x) = ', w0, ' + ', w1, ' * x')
print('Error', error_by_me)


# %% md
# Learning by tool with bi-variate regression.

# %% py
def plot_3d_data(x1_train, x2_train, y_train,
                 x1Model=None, x2Model=None, yModel=None,
                 x1Test=None, x2Test=None, yTest=None, title=None):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    if x1_train:
        plt.scatter(x1_train, x2_train, y_train, c='r', marker='o', label='train data')
    if x1Model:
        plt.scatter(x1Model, x2Model, yModel, c='b', marker='_', label='learnt model')
    if x1Test:
        plt.scatter(x1Test, x2Test, yTest, c='g', marker='^', label='test data')
    plt.title(title)
    ax.set_xlabel("GPD Capital.")
    ax.set_ylabel("Freedom.")
    ax.set_zlabel("Happiness.")
    plt.legend()
    plt.show()


# %%

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from MyRegressor import MySGDRegression
from sklearn.preprocessing import StandardScaler

plt.style.use({'figure.facecolor': 'white'})


# %%
def load_data(filename, input_variable_name, output_variable_name):
    file = pd.read_csv(filename)
    inputs = []
    for input_variable_name in input_variable_name:
        inputs.append([float(val) for val in file[input_variable_name]])
    output = [float(val) for val in file[output_variable_name]]
    return inputs, output


input_data, output_data = load_data('data/v1_world-happiness-report-2017.csv', ['Economy..GDP.per.Capita.', 'Freedom'],
                                    'Happiness.Score')


# %%

def plotDataHistogram(x, variableName):
    n, bins, patches = plt.hist(x, 10)
    plt.title('Histogram of ' + variableName)
    plt.show()


plotDataHistogram(input_data[0], 'Economy GDP per Capital.')
plotDataHistogram(input_data[1], 'Freedom')
plotDataHistogram(output_data, 'Happiness')


# %%

def plot_for_linearity(inputs, outputs, input_label, output_label, title=""):
    plt.plot(inputs, outputs, 'ro')
    plt.xlabel(input_label)
    plt.ylabel(output_label)
    plt.title(title)
    plt.show()


plot_for_linearity(input_data[0], output_data, "GDA capital", "Happiness", 'GDA vs Happiness')


# %%

def split_in_train_and_test_samples(inputs, outputs):
    np.random.seed(5)
    train_sample_indexes = np.random.choice([i for i in range(len(outputs))], int(0.8 * len(outputs)), replace=False)
    validation_sample_indexes = [i for i in range(len(outputs)) if i not in train_sample_indexes]
    train_inputs = []
    validation_inputs = []
    for inp in inputs:
        train_inputs.append([inp[i] for i in train_sample_indexes])
        validation_inputs.append([inp[i] for i in validation_sample_indexes])
    train_outputs = [outputs[i] for i in train_sample_indexes]
    validation_outputs = [outputs[i] for i in validation_sample_indexes]
    return train_inputs, train_outputs, validation_inputs, validation_outputs


train_inputs, train_outputs, validation_inputs, validation_outputs = split_in_train_and_test_samples(input_data,
                                                                                                     output_data)
plt.plot(train_inputs[0], train_outputs, 'ro', label='Train')
plt.plot(validation_inputs[0], validation_outputs, 'g*', label='Test')
plt.xlabel("GDA per capital")
plt.ylabel("Happiness")
plt.title('Train and test data.')
plt.show()


# %%

def learn_by_tool_single_feature():
    xx = [[feature] for feature in train_inputs[0]]
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor.fit(xx, train_outputs)
    return regressor.intercept_[0], regressor.coef_[0]


w0, w1 = learn_by_tool_single_feature()
print('Lear model by tool returned: f(x) = ', w0, ' + ', w1, ' * x')


# %%

def learn_by_my_code_single_feature():
    xx = [[feature] for feature in train_inputs[0]]
    regressor = MySGDRegression()
    regressor.fit_with_batch(xx, train_outputs)
    error = 0.0
    for real_output, computed_output in zip(validation_outputs, regressor.predict([e] for e in validation_inputs[0])):
        error += (real_output - computed_output) ** 2
    error /= len(validation_outputs)
    return regressor.intercept_, regressor.coef_[0], error


w0, w1, error_by_me = learn_by_my_code_single_feature()
print('Lear model by me returned: f(x) = ', w0, ' + ', w1, ' * x')
print('Error', error_by_me)


# %% py

def plot_3d_data(x1_train, x2_train, y_train,
                 x1Model=None, x2Model=None, yModel=None,
                 x1Test=None, x2Test=None, yTest=None, title=None):
    from mpl_toolkits import mplot3d
    ax = plt.axes(projection='3d')
    if x1_train:
        plt.scatter(x1_train, x2_train, y_train, c='r', marker='o', label='train data')
    if x1Model:
        plt.scatter(x1Model, x2Model, yModel, c='b', marker='_', label='learnt model')
    if x1Test:
        plt.scatter(x1Test, x2Test, yTest, c='g', marker='^', label='test data')
    plt.title(title)
    ax.set_xlabel("GPD Capital.")
    ax.set_ylabel("Freedom.")
    ax.set_zlabel("Happiness.")
    plt.legend()
    plt.show()


plot_3d_data(*train_inputs, train_outputs, title='GPD Capital vs Freedom vs Happiness.')


# %% normalisation

def normalisation(trainData, testData):
    scaler = StandardScaler()
    if not isinstance(trainData[0], list):
        # encode each sample into a list
        trainData = [[d] for d in trainData]
        testData = [[d] for d in testData]

        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data

        # decode from list to raw values
        normalisedTrainData = [el[0] for el in normalisedTrainData]
        normalisedTestData = [el[0] for el in normalisedTestData]
    else:
        scaler.fit(trainData)  # fit only on training data
        normalisedTrainData = scaler.transform(trainData)  # apply same transformation to train data
        normalisedTestData = scaler.transform(testData)  # apply same transformation to test data
    return normalisedTrainData, normalisedTestData


def statisticalNormalisation(features):
    meanValue = sum(features) / len(features)
    stdDevValue = (1 / len(features) * sum([(feat - meanValue) ** 2 for feat in features])) ** 0.5
    normalisedFeatures = [(feat - meanValue) / stdDevValue for feat in features]
    return normalisedFeatures


def normalisation_by_me(train_data, test_data):
    if isinstance(train_data[0], list):
        normalized_train_data = []
        normalized_test_data = []
        for feature in train_data:
            normalized_train_data.append(statisticalNormalisation(feature))
        for feature in test_data:
            normalized_test_data.append(statisticalNormalisation(feature))
        return list(map(list, zip(*normalized_train_data))), list(map(list, zip(*normalized_test_data)))
    return statisticalNormalisation(train_data), statisticalNormalisation(test_data)


# print(normalisation(train_inputs, validation_inputs))
train_inputs_for_normalisation = [[feat1, feat2] for feat1, feat2 in zip(*train_inputs)]
validation_inputs_for_normalisation = [[feat1, feat2] for feat1, feat2 in zip(*validation_inputs)]
train_inputs_normalized, validation_inputs_normalized = normalisation(train_inputs_for_normalisation,
                                                                      validation_inputs_for_normalisation)
train_outputs_normalized, validation_outputs_normalized = normalisation(train_outputs, validation_outputs)

train_inputs_normalized, validation_inputs_normalized = normalisation_by_me(train_inputs, validation_inputs)
train_outputs_normalized, validation_outputs_normalized = normalisation_by_me(train_outputs, validation_outputs)

# %%

feature1train = [ex[0] for ex in train_inputs_normalized]
feature2train = [ex[1] for ex in train_inputs_normalized]


# plot_3d_data(
#     feature1train,
#     feature2train,
#     train_outputs_normalized)
# #     ,
#     None,
#     None,
#     None,
#     list([val[0] for val in validation_inputs_normalized]),
#     list([val[1] for val in validation_inputs_normalized]),
#     validation_outputs_normalized,
#     'Test and trian data after normalisation.'
# )


# %%

def learn_by_tool_multiple_feature():
    regressor = linear_model.SGDRegressor(alpha=0.01, max_iter=100)
    regressor.fit(train_inputs_normalized, train_outputs_normalized)
    return regressor.intercept_[0], regressor.coef_[0], regressor.coef_[1]


w0, w1, w2 = learn_by_tool_multiple_feature()
print('Lear model by tool returned: f(x) = ', w0, '+', w1, ' * x1 +', w2, ' * x2.')

# %%

from MyRegressor import MySGDRegression


def learn_by_my_code_multiple_features():
    regressor = MySGDRegression()
    regressor.fit_with_batch(train_inputs_normalized, train_outputs_normalized)
    return regressor.intercept_, regressor.coef_[0], regressor.coef_[1]


w0, w1, w2 = learn_by_my_code_multiple_features()
print('Lear model by me returned: f(x) = ', w0, '+', w1, ' * x1 +', w2, ' * x2.')
print('Error', error_by_me)


def learn_by_my_code_multiple_features(train_inputs, train_outputs):
    regressor = MySGDRegression()
    regressor.fit_with_batch(train_inputs, train_outputs)
    return regressor.intercept_, regressor.coef_[0], regressor.coef_[1]


print('Happiness:')
w0, w1, w2 = learn_by_my_code_multiple_features(train_inputs_normalized, train_outputs_normalized)
print('Lear for happiness returned: f(x) = ', w0, '+', w1, ' * x1 +', w2, ' * x2.')
print('Error', error_by_me)

input_data, output_data = load_data('data/v1_world-happiness-report-2017.csv', ['Economy..GDP.per.Capita.', 'Freedom'], 'Family')
train_inputs, train_outputs, validation_inputs, validation_outputs = split_in_train_and_test_samples(input_data, output_data)
train_inputs_for_normalisation = [[feat1, feat2] for feat1, feat2 in zip(*train_inputs)]
validation_inputs_for_normalisation = [[feat1, feat2] for feat1, feat2 in zip(*validation_inputs)]
# train_inputs_normalized, _ = normalisation(train_inputs_for_normalisation, validation_inputs_for_normalisation)
# train_outputs_normalized, _ = normalisation(train_outputs, validation_outputs)

print('Family:')
w0, w1, w2 = learn_by_my_code_multiple_features(train_inputs_normalized, train_outputs_normalized)
print('Lear model for family returned: f(x) = ', w0, '+', w1, ' * x1 +', w2, ' * x2.')
print('Error', error_by_me)


from sklearn.datasets import make_regression
# create datasets
input_for_multiple_output, output_for_multiple_output = make_regression(n_samples=1000, n_features=5, n_targets=2, random_state=1, noise=0.5)

np.random.seed(5)
train_sample_indexes = np.random.choice([i for i in range(len(input_for_multiple_output))], int(0.8 * len(input_for_multiple_output)), replace=False)
validation_sample_indexes = [i for i in range(len(input_for_multiple_output)) if i not in train_sample_indexes]

train_input_for_multiple_output = [input_for_multiple_output[i] for i in train_sample_indexes]
train_output_for_multiple_output = [output_for_multiple_output[i] for i in train_sample_indexes]
validation_input_for_multiple_output = [input_for_multiple_output[i] for i in validation_sample_indexes]
validation_output_for_multiple_output = [output_for_multiple_output[i] for i in validation_sample_indexes]

model = linear_model.LinearRegression()
model.fit(train_input_for_multiple_output, train_output_for_multiple_output)

plt.plot(*list(map(list, zip(*validation_output_for_multiple_output))), 'ro')
plt.plot(*list(map(list, zip(*model.predict(validation_input_for_multiple_output)))), 'g*')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

print(model.intercept_)
print(model.coef_)
func = lambda x1, x2, x3, x4, x5, index : model.intercept_[index] + model.coef_[index][0] * x1 + model.coef_[index][1] * x2 + model.coef_[index][2] * x3 + model.coef_[index][3] * x4 + model.coef_[index][4] * x5
x = np.linspace(-5, 5, 200)
y = [func(*elems, 1) for elems in validation_input_for_multiple_output]

error = 0
for tr, pred in zip(validation_output_for_multiple_output, model.predict(validation_input_for_multiple_output)):
    error += (tr - pred) ** 2

print("The error is: {0}".format(str(error)))


