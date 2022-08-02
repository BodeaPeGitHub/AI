from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn import neural_network
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import ANN

iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


def load_data_iris(filename):
    header = ['Sepal length', 'Sepal width', "Petal length", "Petal width", "Class"]
    csv_file = pd.read_csv(filename, header=None, names=header)
    columns = [csv_file[name] for name in header]
    inputs = [list(inp) for inp in zip(*columns[:-1])]
    outputs = [iris_classes.index(name) for name in columns[-1]]
    return inputs, outputs


def plot_5d(inputs, outputs, title="Plot for data distribution"):
    x = [inp[0] for inp in inputs]
    y = [inp[1] for inp in inputs]
    z = np.array([inp[2] for inp in inputs])
    v = [inp[3] for inp in inputs]
    figure = px.scatter_3d(width=800,
                           height=700,
                           x=x,
                           y=y,
                           z=z,
                           color=outputs,
                           symbol=v,
                           title=title,
                           labels=dict(x='Sepal length', y="Sepal width", z="Petal length", symbol="Petal width",
                                       color="Flower types"))
    figure.update_layout(legend=dict(
        orientation="v",
        yanchor='top',
        xanchor="right"))
    figure.show()


def split_into_test_and_train_data(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validation_sample = [i for i in indexes if not i in train_sample]
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


def train_model_by_tool(train_inputs, train_outputs):
    temp_classifier = neural_network.MLPClassifier()
    temp_classifier.fit(train_inputs, train_outputs)
    return temp_classifier


def train_model_by_me(train_inputs: np.array, train_outputs: np.array, snow_print=True):
    temp_classifier = ANN.NeuralNetwork()
    temp_classifier.fit(np.array(train_inputs), np.array(train_outputs), print_loss_and_epoch=snow_print)
    return temp_classifier


def plotConfusionMatrix(cm, class_names, title=""):
    import itertools
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix ' + title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    text_format = 'd'
    thresh = cm.max() / 2.
    for row, column in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(column, row, format(cm[row, column], text_format),
                 horizontalalignment='center',
                 color='white' if cm[row, column] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def evalMultiClass(real_labels, computed_labels, labels_names):
    real_labels = list(real_labels)
    computed_labels = list(computed_labels)
    confusion_mat = confusion_matrix(real_labels, computed_labels)
    acc = sum([confusion_mat[i][i] for i in range(len(labels_names))]) / len(real_labels)
    prec = {}
    rec = {}
    for i in range(len(labels_names)):
        prec[labels_names[i]] = confusion_mat[i][i] / sum([confusion_mat[j][i] for j in range(len(labels_names))])
        rec[labels_names[i]] = confusion_mat[i][i] / sum([confusion_mat[i][j] for j in range(len(labels_names))])
    return acc, prec, rec, confusion_mat


if __name__ == '__main__':
    input_data, output_data = load_data_iris('data/iris/iris.data')
    # plot_5d(input_data, output_data)
    train_input, train_output, validation_input, validation_output = split_into_test_and_train_data(input_data, output_data)
    train_input, validation_input = normalisation(train_input, validation_input)
    # plot_5d(train_input, train_output)
    model = train_model_by_tool(train_input, train_output)
    prediction = model.predict(validation_input)
    accuracy, precision, recall, matrix = evalMultiClass(np.array(validation_output), prediction, iris_classes)
    plotConfusionMatrix(matrix, iris_classes, "Iris classification.")
    print("-" * 50 + "Tool classifier" + '-' * 50)
    print('Accuracy:', accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    plotConfusionMatrix(matrix, iris_classes, "Iris classification with my classifier.")
    model = train_model_by_me(train_input, train_output)
    prediction = model.predict(validation_input)
    accuracy, precision, recall, matrix = evalMultiClass(list(validation_output), list(prediction), iris_classes)
    print("-" * 50 + "My classifier" + '-' * 50)
    print('Accuracy:', accuracy)
    print("Precision:", precision)
    print("Recall:", recall)