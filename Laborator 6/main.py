import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt, log, exp


def prediction_performance_for_regression(real_output, predicted_output):
    return sum([abs(real - predict) for real, predict in zip(real_output, predicted_output)]) / len(real_output)


def prediction_performance_for_regression_with_sqrt(real_output, predicted_output):
    return sum([(real - predict) ** 2 for real, predict in zip(real_output, predicted_output)]) / len(real_output)


def plot_points(sport):
    figure = plt.figure(figsize=(6, 6))
    ax = figure.add_subplot(111, projection='3d')
    for x, y, z in zip(sport['Weight'], sport['Waist'], sport['Pulse']):
        ax.scatter(x, y, z, color='b')
    for x, y, z in zip(sport['PredictedWeight'], sport['PredictedWaist'], sport['PredictedPulse']):
        ax.scatter(x, y, z, color='r')
    plt.show()


def mean_absolute_error(file):
    sport = pd.read_csv(file)
    return prediction_performance_for_regression(sport['Weight'],
                                                 sport['PredictedWeight']) + prediction_performance_for_regression(
        sport['Waist'], sport['PredictedWaist']) + prediction_performance_for_regression(sport['Pulse'],
                                                                                         sport['PredictedPulse'])


def root_mean_absolute_error(file):
    sport = pd.read_csv(file)
    return sqrt(prediction_performance_for_regression_with_sqrt(sport['Weight'],
                                                                sport[
                                                                    'PredictedWeight']) + prediction_performance_for_regression_with_sqrt(
        sport['Waist'], sport['PredictedWaist']) + prediction_performance_for_regression_with_sqrt(sport['Pulse'],
                                                                                                   sport[
                                                                                                       'PredictedPulse']))


def evaluate_classification(file):
    flowers = pd.read_csv(file)
    real_values = flowers['Type']
    predicted_values = flowers['PredictedType']
    labels = set(real_values)
    accuracy = sum([1 if real == predict else 0 for real, predict in zip(real_values, predicted_values)]) / len(
        real_values)
    true_positive = {}
    false_positive = {}
    true_negative = {}
    false_negative = {}
    for label in labels:
        true_positive[label] = sum(
            [1 if real == predict == label else 0 for real, predict in zip(real_values, predicted_values)])
        false_positive[label] = sum(
            [1 if real != predict == label else 0 for real, predict in zip(real_values, predicted_values)])
        true_negative[label] = sum(
            [1 if real != label != predict else 0 for real, predict in zip(real_values, predicted_values)])
        false_negative[label] = sum(
            [1 if real == label != predict else 0 for real, predict in zip(real_values, predicted_values)])
    precision = {}
    recall = {}
    for label in labels:
        precision[label] = true_positive[label] / (true_positive[label] + false_positive[label])
        recall[label] = true_positive[label] / (true_positive[label] + false_negative[label])

    return accuracy, precision, recall


def compute_loss_for_regression(real_output, predicted_output):
    return sum([abs(real - predict) for real, predict in zip(real_output, predicted_output)])


def evaluate_log_loss(real_values, predicted_values):
    real_outputs = [[1, 0] if label == 'spam' else [0, 1] for label in real_values]
    return sum([-sum(
        [real_probability * log(predict_probability) for real_probability, predict_probability in zip(real, predict)])
                for real, predict in zip(real_outputs, predicted_values)]) / len(real_values)


def evaluate_softmax_loss(real_values, raw_outputs):
    # multiclass - nu se poate repeta
    # apply softmax for all raw outputs
    expected_values = [exp(val) for val in raw_outputs]
    mapped_values = [val / sum(expected_values) for val in expected_values]
    print(mapped_values, ' sum: ', sum(mapped_values))
    return -sum([real * log(mapped) for real, mapped in zip(real_values, mapped_values)])


def evaluate_sigmoid_loss(real_values, raw_outputs):
    # multi label - poate fi mai multe.
    # apply softmax for all raw outputs
    mapped_outputs = [1 / (1 + exp(-val)) for val in raw_outputs]
    print(mapped_outputs, ' sum: ', sum(mapped_outputs))
    return -sum([real * log(mapped) for real, mapped in zip(real_values, mapped_outputs)])


print('Mean absolute error:', mean_absolute_error('sport.csv'))
print('Root mean square error:', root_mean_absolute_error('sport.csv'))
plot_points(pd.read_csv('sport.csv'))
accuracy, precision, recall = evaluate_classification('flowers.csv')
print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('CE Loss:', evaluate_log_loss(['spam', 'spam', 'ham', 'ham', 'spam', 'ham'],
                                    [[0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.9, 0.1], [0.7, 0.3], [0.4, 0.6]]))
print(evaluate_sigmoid_loss([0, 1, 0, 0, 1], [-0.5, 1.2, 0.1, 2.4, 0.3]))
print(evaluate_softmax_loss([0, 1, 0, 0, 0], [-0.5, 1.2, 0.1, 2.4, 0.3]))
