import numpy as np


# Implemented with the help of: https://stackabuse.com/creating-a-neural-network-from-scratch-in-python-multi-class-classification/


class NeuralNetwork:

    def __init__(self, hidden_layer_sizes=(10,), activation='relu', max_iter=5000, learning_rate_init=0.001):
        self.__weights = []
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation = activation
        self.__max_iter = max_iter
        self.__learning_rate_init = learning_rate_init

    @staticmethod
    def __softmax(vector: np.array):
        exp_vector = np.exp(vector)
        return exp_vector / exp_vector.sum(axis=1, keepdims=True)

    @staticmethod
    def __sigmoid(vector: np.array):
        return 1 / (1 + np.exp(-vector))

    def __sigmoid_derivate(self, vector: np.array):
        return self.__sigmoid(vector) * (1 - self.__sigmoid(vector))

    @staticmethod
    def __transform_output_data(output_data, no_of_output_labels):
        new_output_data = np.zeros((len(output_data), no_of_output_labels))
        for i in range(len(output_data)):
            new_output_data[i, output_data[i]] = 1
        return new_output_data

    def fit(self, input_data, output_data, print_loss_and_epoch=True, number_of_epoch_to_print=50):
        no_of_output_labels = len(set(output_data))
        output_data = self.__transform_output_data(output_data, no_of_output_labels)
        if len(input_data) == 0:
            raise Exception("Input data must have at least 1 sample.")
        no_input_attributes = len(input_data[0])
        hidden_nodes = self.__hidden_layer_sizes[0]
        # Weight matrix for input layer X hidden layer
        wh = np.random.rand(no_input_attributes, hidden_nodes)
        # Biases for first matrix
        bh = np.random.randn(hidden_nodes)
        # Weight matrix for hidden layer X output layer
        wo = np.random.rand(hidden_nodes, no_of_output_labels)
        # Biases for second matrix
        bo = np.random.randn(no_of_output_labels)

        for epoch in range(self.__max_iter):
            # Calculate the output
            zh = np.dot(input_data, wh) + bh  # zh -
            ah = self.__sigmoid(zh)
            zo = np.dot(ah, wo) + bo
            ao = self.__softmax(zo)

            # Back propagation
            cost = ao - output_data
            cost_wo = np.dot(ah.T, cost)
            cost_bo = cost

            cost_dah = np.dot(cost, wo.T)
            dah_dzh = self.__sigmoid_derivate(zh)
            dzh_dwh = input_data
            cost_wh = np.dot(dzh_dwh.T, dah_dzh * cost_dah)
            cost_bh = cost_dah * dah_dzh

            wh -= self.__learning_rate_init * cost_wh
            bh -= self.__learning_rate_init * cost_bh.sum(axis=0)
            wo -= self.__learning_rate_init * cost_wo
            bo -= self.__learning_rate_init * cost_bo.sum(axis=0)
            if epoch % number_of_epoch_to_print == 0 and print_loss_and_epoch:
                loss = np.sum(-output_data * np.log(ao))
                print('Epoch ' + str(epoch) + '. Loss: ' + str(loss))

        self.__weights = [
            wh,
            bh,
            wo,
            bo
        ]

    def predict(self, input_data):
        wh, bh, wo, bo = self.__weights
        zh = np.dot(input_data, wh) + bh
        ah = self.__sigmoid(zh)
        zo = np.dot(ah, wo) + bo
        return list(map(lambda x: list(x).index(max(x)), zo))
