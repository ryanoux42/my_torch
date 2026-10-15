#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## my_pgp
## File description:
## python
##

import random

def initialize_perceptron():
    weight1 = random.uniform(-1, 1)
    weight2 = random.uniform(-1, 1)
    bias = random.uniform(-1, 1)
    learning_rate = 0.1
    return weight1, weight2, bias, learning_rate


def activation_function(weighted_sum):
    return 1 if weighted_sum >= 0 else 0


def check_accuracy(weight1, weight2, bias, data):
    for inputs, expected_output in data:
        weighted_sum = inputs[0] * weight1 + inputs[1] * weight2 + bias
        output = activation_function(weighted_sum)
        if output != expected_output:
            return False
    return True


def train_perceptron(weight1, weight2, bias, learning_rate, data, max_iterations=100):
    for iteration in range(max_iterations):
        random.shuffle(data)
        for inputs, expected_output in data:
            weighted_sum = inputs[0] * weight1 + inputs[1] * weight2 + bias
            output = activation_function(weighted_sum)
            error = expected_output - output
            if error != 0:
                weight1 += learning_rate * error * inputs[0]
                weight2 += learning_rate * error * inputs[1]
                bias += learning_rate * error
    return weight1, weight2, bias


def main():
    data = [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ]
    weight1, weight2, bias, learning_rate = initialize_perceptron()
    while True:
        print("Starting accuracy check...")
        if check_accuracy(weight1, weight2, bias, data):
            print("Perceptron is valid. Training is complete!")
            break
        else:
            print("Accuracy is incorrect. Starting training...")
            weight1, weight2, bias = train_perceptron(weight1, weight2, bias, learning_rate, data, 100)
    print(f"Final Weights: Weight1 = {weight1:.2f}, Weight2 = {weight2:.2f}, Bias = {bias:.2f}")
    if check_accuracy(weight1, weight2, bias, data):
        print("Final perceptron output is correct for all inputs.")
    else:
        print("There was an error in the final output.")

if __name__ == "__main__":
    main()
