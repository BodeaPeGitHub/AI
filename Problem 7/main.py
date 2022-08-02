import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def load_data(filename, input_variable_name, output_variable_name):
    file = pd.read_csv(filename)
    inputs = []
    for input_variable_name in input_variable_name:
        inputs.append([float(val) for val in file[input_variable_name]])
    output = [float(val) for val in file[output_variable_name]]
    return inputs, output


def plot_data_histogram(data, variable_name):
    n, bins, patches = plt.hist(data, 10)
    plt.title("Histogram of " + variable_name)
    plt.show()


def plot_to_check_linearity(input, input_name, output, output_name):
    plt.plot(input, output, 'ro')
    plt.xlabel(input_name)
    plt.ylabel(output_name)
    plt.title(input_name + " vs " + output_name)
    plt.show()


def train_and_test_samples(inputs, outputs):
    np.random.seed(6)
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




# clear data
def clear_data(inputs, outputs):
    set_data = []
    for index, elems in enumerate(zip(inputs[0], inputs[1])):
        inp1, inp2 = elems
        if pd.isna(inp1) or pd.isna(inp2) or inp1 == inp2 == 0 or [inp1, inp2] in set_data:
            inputs[0].pop(index)
            inputs[1].pop(index)
            outputs.pop(index)
        else:
            set_data.append([inp1, inp2])
    return inputs, outputs


# plot_data_histogram(inputs[0], 'Economy..')
# plot_data_histogram(inputs[1], 'Freedom..')
# plot_data_histogram(outputs, 'Final..')




def linear_regression_by_tool(train_inputs, train_outputs):
    xx = [[x, y] for x, y in zip(train_inputs[0], train_inputs[1])]
    regressor = linear_model.LinearRegression()
    regressor.fit(xx, train_outputs)
    return regressor


# predict_outputs = regressor.predict([[x, y] for x, y in zip(validation_inputs[0], validation_inputs[1])])
#
# plt.plot(validation_inputs[0], predict_outputs, 'yo')
# plt.plot(validation_inputs[0], validation_outputs, 'g^')
# plt.show()
#
#
# error = mean_squared_error(validation_outputs, predict_outputs)
# print(error)


def linear_regression_by_me(inputs, outputs):
    XT = [[1] * len(outputs)] + inputs
    XTX = []
    for row1 in XT:
        line = []
        for row2 in XT:
            line.append(sum([x * y for x, y in zip(row1, row2)]))
        XTX.append(line)
    XTX_inverse = get_matrix_inverse(XTX)
    XTX_inverse_XT = []
    for row in XTX_inverse:
        line = []
        for row1 in transpose_matrix(XT):
            line.append(sum([x * y for x, y in zip(row, row1)]))
        XTX_inverse_XT.append(line)
    XTX_inverse_XTY = []
    for row in XTX_inverse_XT:
        XTX_inverse_XTY.append(sum([x * y for x, y in zip(row, outputs)]))
    return XTX_inverse_XTY


def transpose_matrix(m):
    return list(map(list, zip(*m)))


def get_matrix_minor(m, i, j):
    return [row[:j] + row[j + 1:] for row in (m[:i] + m[i + 1:])]


def get_matrix_determinant(m):
    # base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1) ** c) * m[0][c] * get_matrix_determinant(get_matrix_minor(m, 0, c))
    return determinant


def get_matrix_inverse(m):
    determinant = get_matrix_determinant(m)
    print('Determinant: ', determinant)
    # special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1] / determinant, -1 * m[0][1] / determinant],
                [-1 * m[1][0] / determinant, m[0][0] / determinant]]

    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = get_matrix_minor(m, r, c)
            cofactorRow.append(((-1) ** (r + c)) * get_matrix_determinant(minor))
        cofactors.append(cofactorRow)
    cofactors = transpose_matrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c] / determinant
    return cofactors


inputs, outputs = load_data('data/v3_world-happiness-report-2017.csv', ['Economy..GDP.per.Capita.', 'Freedom'],
                            'Happiness.Score')
inputs, outputs = clear_data(inputs, outputs)
train_inputs, train_outputs, validation_inputs, validation_outputs = train_and_test_samples(inputs, outputs)
regressor = linear_regression_by_tool(train_inputs, train_outputs)

print('By tool', [] + [regressor.intercept_] + list(regressor.coef_))
print('By me', linear_regression_by_me(train_inputs, train_outputs))
