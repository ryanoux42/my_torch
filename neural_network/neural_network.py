#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## my_pgp
## File description:
## python
##

import numpy as np

class Neural_Network:
    def __init__(self, config):
        self.input_nb = config["input_nb"]
        self.hidden_layer_1_nb = config["hidden_layer_1_nb"]
        self.hidden_layer_2_nb = config["hidden_layer_2_nb"]
        self.hidden_layer_3_nb = config["hidden_layer_3_nb"]
        self.output_nb = config["output_nb"]
        self.learning_rate = config["learning_rate"]
        self.dropout_rate = config["dropout"]
        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.lambda_ = config["lambda"]
        self.epsilon = 1e-8
        self.weights = {
            "W1": np.random.randn(self.hidden_layer_1_nb, self.input_nb) * np.sqrt(2 / self.input_nb),
            "W2": np.random.randn(self.hidden_layer_2_nb, self.hidden_layer_1_nb) * np.sqrt(2 / self.hidden_layer_1_nb),
            "W3": np.random.randn(self.hidden_layer_3_nb, self.hidden_layer_2_nb) * np.sqrt(2 / self.hidden_layer_2_nb),
            "W4": np.random.randn(self.output_nb, self.hidden_layer_3_nb) * np.sqrt(2 / self.hidden_layer_3_nb),
        }
        self.biases = {
            "b1": np.zeros((self.hidden_layer_1_nb, 1)),
            "b2": np.zeros((self.hidden_layer_2_nb, 1)),
            "b3": np.zeros((self.hidden_layer_3_nb, 1)),
            "b4": np.zeros((self.output_nb, 1)),
        }
        self.m = {key: np.zeros_like(value) for key, value in self.weights.items()}
        self.v = {key: np.zeros_like(value) for key, value in self.weights.items()}
        self.t = 1
        self.epochs = config["epochs"]

    def leaky_relu(self, z, alpha=0.01):
        return np.maximum(alpha * z, z)

    def leaky_relu_derivative(self, z, alpha=0.01):
        return np.where(z > 0, 1, alpha)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward_propagation(self, X, training):
        Z1 = np.dot(self.weights["W1"], X) + self.biases["b1"]
        A1 = self.leaky_relu(Z1)
        if training:
            A1 *= np.random.binomial([np.ones_like(A1)], 1 - self.dropout_rate)[0] / (1 - self.dropout_rate)

        Z2 = np.dot(self.weights["W2"], A1) + self.biases["b2"]
        A2 = self.leaky_relu(Z2)
        if training:
            A2 *= np.random.binomial([np.ones_like(A2)], 1 - self.dropout_rate)[0] / (1 - self.dropout_rate)

        Z3 = np.dot(self.weights["W3"], A2) + self.biases["b3"]
        A3 = self.leaky_relu(Z3)
        if training:
            A3 *= np.random.binomial([np.ones_like(A3)], 1 - self.dropout_rate)[0] / (1 - self.dropout_rate)

        Z4 = np.dot(self.weights["W4"], A3) + self.biases["b4"]
        A4 = self.sigmoid(Z4)

        cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3, "A3": A3, "Z4": Z4, "A4": A4}
        return A4, cache

    def backward_propagation(self, X, Y, cache):
        m = X.shape[1]
        gradients = {}

        A4 = cache["A4"]
        dZ4 = A4 - Y
        gradients["dW4"] = np.dot(dZ4, cache["A3"].T) / m + (self.lambda_ * self.weights["W4"]) / m
        gradients["db4"] = np.sum(dZ4, axis=1, keepdims=True) / m

        dA3 = np.dot(self.weights["W4"].T, dZ4)
        dZ3 = dA3 * self.leaky_relu_derivative(cache["Z3"])
        gradients["dW3"] = np.dot(dZ3, cache["A2"].T) / m + (self.lambda_ * self.weights["W3"]) / m
        gradients["db3"] = np.sum(dZ3, axis=1, keepdims=True) / m

        dA2 = np.dot(self.weights["W3"].T, dZ3)
        dZ2 = dA2 * self.leaky_relu_derivative(cache["Z2"])
        gradients["dW2"] = np.dot(dZ2, cache["A1"].T) / m + (self.lambda_ * self.weights["W2"]) / m
        gradients["db2"] = np.sum(dZ2, axis=1, keepdims=True) / m

        dA1 = np.dot(self.weights["W2"].T, dZ2)
        dZ1 = dA1 * self.leaky_relu_derivative(cache["Z1"])
        gradients["dW1"] = np.dot(dZ1, X.T) / m + (self.lambda_ * self.weights["W1"]) / m
        gradients["db1"] = np.sum(dZ1, axis=1, keepdims=True) / m

        return gradients


    def update_parameters(self, gradients):
        for key in self.weights.keys():

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradients[f"d{key}"]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * np.square(gradients[f"d{key}"])
            m_corrected = self.m[key] / (1 - self.beta1**self.t)
            v_corrected = self.v[key] / (1 - self.beta2**self.t)
            self.weights[key] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + self.epsilon)
            self.biases[f"b{key[-1]}"] -= self.learning_rate * gradients[f"db{key[-1]}"]

        self.t += 1

    def create_batches(self, X, Y, batch_size):
        m = X.shape[1]
        permutation = np.random.permutation(m)
        X_shuffled = X[:, permutation]
        Y_shuffled = Y[:, permutation]
        for i in range(0, m, batch_size):
            yield X_shuffled[:, i:i + batch_size], Y_shuffled[:, i:i + batch_size]

    def adjust_learning_rate(self, epoch):
        if epoch % 200 == 0 and epoch > 0:
            self.learning_rate *= 0.5

    def train(self, X, Y):
        threshold = 0.5
        epoch_counter = 0
        for epoch in range(self.epochs):
            batch_counter = 0
            for X_batch, Y_batch in self.create_batches(X, Y, batch_size=100):
                A4, cache = self.forward_propagation(X_batch, True)
                prediction = (A4 >= threshold).astype(int)
                loss = -np.mean(Y_batch * np.log(A4 + 1e-8))
                l2_penalty = (self.lambda_ / (2 * X_batch.shape[1])) * sum([np.sum(np.square(self.weights[key])) for key in self.weights.keys()])
                loss += l2_penalty
                gradients = self.backward_propagation(X_batch, Y_batch, cache)
                self.update_parameters(gradients)
                self.adjust_learning_rate(epoch)
                if batch_counter % 100 == 0:
                    print(f"Sample predictions: {A4[:, :5]}")
                    print(f"Sample labels: {Y_batch[:, :5]}")
                    print(f"Epoch {epoch}, Batch {batch_counter}: Loss = {loss:.4f}, Prediction Accuracy: {np.mean(prediction == Y_batch)}")
                batch_counter += 1
            epoch_counter += 1
        print(f"Training completed for {epoch_counter} epochs.")
