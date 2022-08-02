from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import neural_network
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import KMeans

iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def load_data_iris(filename, header=None):
    if header is None:
        header = ['Sepal length', 'Sepal width', "Petal length", "Petal width", "Class"]
    csv_file = pd.read_csv(filename, header=None, names=header)
    columns = [csv_file[name] for name in header]
    inputs = [list(inp) for inp in zip(*columns[:-1])]
    outputs = [iris_classes.index(name) for name in columns[-1]]
    return inputs, outputs


def split_into_test_and_train_data(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validation_sample = [i for i in indexes if i not in train_sample]
    train_input_data = [inputs[i] for i in train_sample]
    train_output_data = [outputs[i] for i in train_sample]
    test_input_data = [inputs[i] for i in validation_sample]
    test_output_data = [outputs[i] for i in validation_sample]
    return train_input_data, train_output_data, test_input_data, test_output_data


# Important! Use the same scaler for train and test data.
def normalisation(train_data, test_data):
    scaler = StandardScaler()
    if not isinstance(train_data[0], list):
        # for one feature only
        train_data = [[row] for row in train_data]
        test_data = [[row] for row in test_data]
        scaler.fit(train_data)
        normalized_train_data = [row[0] for row in scaler.transform(train_data)]
        normalized_test_data = [row[0] for row in scaler.transform(test_data)]
        return normalized_train_data, normalized_test_data
    # for input with more than one feature
    scaler.fit(train_data)
    normalized_train_data = scaler.transform(train_data)
    normalized_test_data = scaler.transform(test_data)
    return normalized_train_data, normalized_test_data


if __name__ == '__main__':
    input_data, output_data = load_data_iris('data/iris/iris.data', header=['Sepal length', 'Sepal width', 'Class'])
    train_input, train_output, validation_input, validation_output = split_into_test_and_train_data(input_data,
                                                                                                    output_data)
    train_input, validation_input = normalisation(train_input, validation_input)
    x = np.array([i[0] for i in train_input])
    y = np.array([i[1] for i in train_input])
    plt.plot(x, y)
    # plt.xlabel("Sepal length")
    # plt.ylabel("Sepal width")
    # plt.title("Iris data 2 features.")
    # plt.show()

