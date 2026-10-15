#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## my_pgp
## File description:
## python
##

import sys
import numpy as np
import pickle
from neural_network.neural_network import Neural_Network

class Analyze:
    def __init__(self):
        self.piece_map = {
            'p': -1,
            'n': -2,
            'b': -3,
            'r': -4,
            'q': -5,
            'k': -6,
            'P': 1,
            'N': 2,
            'B': 3,
            'R': 4,
            'Q': 5,
            'K': 6,
        }
        self.chessboards = []
        self.predict = True
        self.savefile = ""

    def convert_fen_to_board(self, fen):
        parts = fen.split()
        fen_board = parts[0]
        board = np.zeros(64)
        ranks = fen_board.split('/')
        square_index = 0
        for rank in ranks:
            for char in rank:
                if char.isdigit():
                    square_index += int(char)
                else:
                    board[square_index] = self.piece_map[char]
                    square_index += 1
        return board

    def get_expected_output(self, fen):
        parts = fen.split()
        expected_output = parts[-1]
        if expected_output == "Nothing":
            return 0
        else:
            return 1
        return expected_output

    def split_data(self, chessboards, train_ratio=0.8):
        np.random.shuffle(chessboards)
        boards = []
        labels = []
        for chessboard in chessboards:
            chessboard = chessboard.strip()
            boards.append(self.convert_fen_to_board(chessboard))
            labels.append(self.get_expected_output(chessboard))
        boards = np.array(boards)
        labels = np.array(labels)
        split_index = int(len(boards) * train_ratio)
        X_train = boards[:split_index].T
        Y_train = labels[:split_index].reshape(1, -1)
        X_test = boards[split_index:].T
        Y_test = labels[split_index:].reshape(1, -1)
        return X_train, X_test, Y_train, Y_test

    def run(self, neural_network):
        if self.predict:
            boards = np.array([self.convert_fen_to_board(cb.strip()) for cb in self.chessboards]).T
            predictions, _ = neural_network.forward_propagation(boards, False)
            # predictions, _ = neural_network.forward_propagation(boards, True)
            predictions = (predictions >= 0.5).astype(int)
            for pred in predictions.T:
                print("Check" if pred[0] == 1 else "Nothing")
        else:
            X_train, X_test, Y_train, Y_test = self.split_data(self.chessboards)
            neural_network.train(X_train, Y_train)
            # predictions, _ = neural_network.forward_propagation(boards, False)
            predictions, _ = neural_network.forward_propagation(X_test, True)
            predictions = (predictions >= 0.5).astype(int)
            accuracy = np.mean(predictions == Y_test)
            print(f"Test Accuracy: {accuracy * 100:.2f}%")
            # if accuracy > 0.9:
            with open(self.savefile, "wb") as f:
                pickle.dump(neural_network, f)
                print(f"Model saved to {self.savefile}")
        return 0

def load_model(filename):
    try:
        with open(filename, "rb") as f:
            model = pickle.load(f)
            return model
    except Exception as e:
        raise Exception(f"Error loading model from {filename}: {e}")

def load_chessboard(filename):
    try:
        chessboards = []
        with open(filename, "r") as f:
            chessboards = f.readlines()
            return chessboards
    except Exception as e:
        raise Exception(f"Error loading chessboards from {filename}: {e}")

def start_neural_network(analyze, predict, model, chessboard_path):
    try:
        neural_network = load_model(model)
        analyze.chessboards = load_chessboard(chessboard_path)
    except Exception as e:
        print(e)
        return 84
    analyze.predict = predict
    analyze.run(neural_network)
    return 0

def main():
    analyze = Analyze()
    nb_arg = len(sys.argv)
    if nb_arg < 2:
        return 84
    if sys.argv[1] == "--predict" and nb_arg == 4:
        return start_neural_network(analyze, True, sys.argv[-2], sys.argv[-1])
    if sys.argv[1] == "--train" and nb_arg in {4, 6}:
        if nb_arg == 6 and sys.argv[2] != "--save":
            return 84
        analyze.savefile = sys.argv[3] if nb_arg == 6 else sys.argv[2]
        return start_neural_network(analyze, False, sys.argv[-2], sys.argv[-1])
    return 84

if __name__ == "__main__" :
    if main() == 84 :
        exit(84)
