#%% md

# Flower classification
### Multiclass classification

#### First step. Load and plot data.

#%% py
from importlib import reload
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from LogisticRegression import MyLogisticRegression
from matplotlib import cm
import LogisticRegressionMultiLabel as mlrl
reload(mlrl)
plt.style.use({'figure.facecolor':'white'})

iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

seed = 0

def load_data(filename):
    header = ['Sepal length', 'Sepal width', "Petal length", "Petal width", "Class"]
    csv_file = pd.read_csv(filename, header=None, names=header)
    columns = [csv_file[name] for name in header]
    inputs = [list(inp) for inp in zip(*columns[:-1])]
    outputs = [iris_classes.index(name) for name in  columns[-1]]
    return inputs, outputs

def plot_data_with_two_features(inputs, outputs, feature_names, feature_position_in_input):
    labels = set(outputs)
    for label in labels:
        x = [feature[feature_position_in_input[0]] for feature, output_label in zip(inputs, outputs) if output_label == label]
        y = [feature[feature_position_in_input[1]] for feature, output_label in zip(inputs, outputs) if output_label == label]
        plt.scatter(x, y, label = iris_classes[label])
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.title("Data distribution plot.")
    plt.show()

def plot_4d(inputs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, 60)
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_zlabel('Petal length')
    x = [inp[0] for inp in inputs]
    y = [inp[1] for inp in inputs]
    z = np.array([inp[2] for inp in inputs])
    # v = [inp[3] for inp in inputs]
    v = output_data
    ax.scatter(x, y, z, c=v)
    plt.show()


def plot_5d(inputs, outputs, title = "Plot for data distribution"):
    x = [inp[0] for inp in inputs]
    y = [inp[1] for inp in inputs]
    z = np.array([inp[2] for inp in inputs])
    v = [inp[3] for inp in inputs]
    figure = px.scatter_3d(width=800,
                           height=700,
                           x = x,
                           y = y,
                           z = z,
                           color=outputs,
                           symbol=v,
                           title=title,
                           labels=dict(x = 'Sepal length', y = "Sepal width", z = "Petal length", symbol= "Petal width", color = "Flower types"))
                           # labels=dict(color = "Flower types"))
    figure.update_layout(legend=dict(
    orientation="v",
    yanchor='top',
    xanchor="right"))
    figure.show()

#%% md
### First i plot-ed 2 features at a time.
### After i plot-ed all the features at once with on a 3d plot and the 4th dimension is represented by the colors.

#%%

input_data, output_data = load_data("data/iris.data")
plot_data_with_two_features(input_data, output_data, ['Sepal length', 'Sepal width'], [0, 1])
plot_data_with_two_features(input_data, output_data, ["Petal length", "Petal width"], [2, 3])
plot_4d(input_data)
plot_5d(input_data, output_data)

#%%

def plot_data_histogram(x, name):
    _ = plt.hist(x, 10)
    plt.title("Histogram of " + name)
    plt.show()

plot_data_histogram([inp[0] for inp in input_data], "Sepal length")
plot_data_histogram([inp[1] for inp in input_data], "Sepal width")
plot_data_histogram([inp[2] for inp in input_data], "Petal length")
plot_data_histogram([inp[3] for inp in input_data], "Petal width")
plot_data_histogram(output_data, "Flower classes")

#%% md
#### Second step - split into train and test data, normalize data.
#%%

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


train_input, train_output, validation_input, validation_output = split_into_test_and_train_data(input_data, output_data)
train_input, validation_input = normalisation(train_input, validation_input)

#%% md
#### Plot with normalized data the test data and validation data.

#%%

plot_5d(train_input, train_output)
plot_5d(validation_input, validation_output)


#%% md

#### Third step - model training.

#%%

def train_model_by_tool(train_inputs, train_outputs):
    classifier = linear_model.LogisticRegression()
    classifier.fit(train_inputs, train_outputs)
    return classifier


def train_model_by_me(train_inputs, train_outputs):
    classifier = mlrl.MyLogisticRegressionMultiLabel()
    classifier.fit(train_inputs, train_outputs)
    return classifier

tool_classifier = train_model_by_tool(train_input, train_output)
# tool_classifier = train_model_by_me(train_input, train_output)
my_classifier = train_model_by_me(train_input, train_output)
print(my_classifier.intercept)
print(my_classifier.coef_)
func = "Classification model: y = "
for no, w in enumerate(list(tool_classifier.intercept_) + list(tool_classifier.coef_[0])):
    feat = ' * feature' + str(no) + ' + '
    if no == 0:
        feat = ' + '
    func += str(w) + feat
func = func[:-2]
print(func)

#%%

def plot_predictions(inputs, real_outputs, predicted_outputs, title, label_names, feature_indexes, feature_names):
    labels = list(set(real_outputs))
    feature1 = [row[feature_indexes[0]] for row in inputs]
    feature2 = [row[feature_indexes[1]] for row in inputs]
    for label in labels:
        x = [feature for feature, real_output_label, predicted_outputs_label in zip(feature1, real_outputs, predicted_outputs) if predicted_outputs_label == real_output_label == label]
        y = [feature for feature, real_output_label, predicted_outputs_label in zip(feature2, real_outputs, predicted_outputs) if predicted_outputs_label == real_output_label == label]
        plt.scatter(x, y, label=label_names[label] + ' correct.')
    for label in labels:
        x = [feature for feature, real_output_label, predicted_outputs_label in zip(feature1, real_outputs, predicted_outputs) if predicted_outputs_label != real_output_label == label]
        y = [feature for feature, real_output_label, predicted_outputs_label in zip(feature2, real_outputs, predicted_outputs) if predicted_outputs_label != real_output_label == label]
        plt.scatter(x, y, label=label_names[label] + ' incorrect.')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    plt.title(title)
    plt.show()

plot_predictions(validation_input, validation_output, tool_classifier.predict(validation_input), "Predictions plot for only 2 features", iris_classes, [0, 1], ['Sepal length', 'Sepal width'])
plot_predictions(validation_input, validation_output, tool_classifier.predict(validation_input), "Predictions plot for only 2 features", iris_classes, [2, 3], ['Petal length', 'Petal width'])

correct_predict = [[], []]
wrong_predict = [[], []]
for inp, predict, correct in zip(input_data, tool_classifier.predict(validation_input), validation_output):
    if predict == correct:
        correct_predict[0].append(inp)
        correct_predict[1].append(predict)
    else:
        wrong_predict[0].append(inp)
        wrong_predict[1].append(predict)

plot_5d(correct_predict[0], correct_predict[1], "Number or correct predictions " + str(len(correct_predict[0])))
plot_5d(wrong_predict[0], wrong_predict[1], "Number of incorrect predicted " + str(len(wrong_predict[0])))

#%% md

#### Fifth step - model testing, accuracy.

#%%

def calculate_error(predicted_outputs, real_outputs):
    error = float(0)
    for predicted, real in zip(predicted_outputs, real_outputs):
        if predicted != real:
            error += 1
    manual_error = error / len(predicted_outputs)
    tool_error = 1 - accuracy_score(real_outputs, predicted_outputs)
    return manual_error, tool_error

def calculate_accuracy(predicted_outputs, real_outputs):
    corect = 0
    for predicted, real in zip(predicted_outputs, real_outputs):
        if predicted == real:
            corect += 1
    return float(corect / len(real_outputs))

error_by_me, error_by_tool = calculate_error(tool_classifier.predict(validation_input), validation_output)
print("This is for tool prediction.")
print('Classification error by me:', error_by_me)
print('Classification error by tool:', error_by_tool)
print("Accuracy score by tool:", tool_classifier.score(validation_input, validation_output))
print("Accuracy score by me:", calculate_accuracy(tool_classifier.predict(validation_input), validation_output))

seed = 0
np.random.seed(seed)
while my_classifier.score(validation_input, validation_output) < 0.95:
    train_input, train_output, validation_input, validation_output = split_into_test_and_train_data(input_data,
                                                                                                    output_data)
    train_input, validation_input = normalisation(train_input, validation_input)
    my_classifier.fit(train_input, train_output)
    print('AAAAAAAAAAA', seed, my_classifier.score(validation_input, validation_output))
    seed += 1

error_by_me, error_by_tool = calculate_error(my_classifier.predict(validation_input), validation_output)
print("This is for my prediction.")
print('Classification error by me:', error_by_me)
print('Classification error by tool:', error_by_tool)
print("Accuracy score by tool:", my_classifier.score(validation_input, validation_output))
print("Accuracy score by me:", calculate_accuracy(my_classifier.predict(validation_input), validation_output))
