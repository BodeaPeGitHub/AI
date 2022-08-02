import pandas as pd
import numpy as np


def load_data(filename, input_variable_name, output_variable_name):
    file = pd.read_csv(filename)
    inputs = [val for val in file[input_variable_name]]
    output = [val for val in file[output_variable_name]]
    return inputs, output


def aplit_data_into_train_and_test(inputs, outputs):
    indexes = [i for i in range(len(inputs))]
    train_sample = np.random.choice(indexes, int(0.8 * len(indexes)), replace=False)
    test_sample = [i for i in indexes if i not in train_sample]

    train_inputs = [inputs[i] for i in train_sample]
    train_outputs = [outputs[i] for i in train_sample]
    test_inputs = [inputs[i] for i in test_sample]
    test_outputs = [outputs[i] for i in test_sample]
    return train_inputs, train_outputs, test_inputs, test_outputs


input_data, output_data = load_data("data/spam.csv", "emailText", "emailType")
train_input_data, train_output_data, validation_input_data, validation_output_data = aplit_data_into_train_and_test(input_data, output_data)
for email, email_type in zip(train_input_data, train_output_data[:10]):
    print(email + " " * (200 - len(email)) + " | " + email_type)
print("-" * 70 + " Validation data " + "-" * 70)
for email, email_type in zip(validation_input_data[:10], validation_output_data):
    print(email + " " * (200 - len(email)) + " | " + email_type)
