#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## my_pgp
## File description:
## python
##

import sys
import json
import pickle
from neural_network.neural_network import Neural_Network

class Generator:
    def __init__(self):
        self.configs = []

    def add_config_nn(self, config_file_i_path, nb_i):
        try:
            nb_i = int(nb_i)
        except ValueError:
            print(f"Error: {nb_i} not int")
            return 84
        try:
            with open(config_file_i_path, "r", encoding="utf-8") as config_file_i:
                data = json.load(config_file_i)
                config_file_i_path = config_file_i_path[:-5]
                self.configs.append((config_file_i_path, data, nb_i))
                return 0
        except FileNotFoundError:
            print(f"Error: {config_file_i_path} not found")
            return 84
        except json.JSONDecodeError:
            print(f"Error: {config_file_i_path} incorrect json")
            return 84

    def run(self):
        for config_file_i_path, config, nb_nn in self.configs:
            for i in range (0, nb_nn):
                file_name = f"{config_file_i_path}_{i + 1}.nn"
                try:
                    with open(file_name, "wb") as f:
                        test = Neural_Network(config)
                        pickle.dump(test , f)
                    print(f"{file_name} created")
                except Exception as e:
                    print(f"Error {file_name}: {e}")

def display_help():
    print("USAGE")
    print("   ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]")
    print("\nDESCRIPTION")
    print("     config_file_i    Configuration file containing description of a neural network we want to generate.")
    print("     nb_i             Number of neural networks to generate based on the configuration file.")

def main() :
    gen = Generator()
    if len(sys.argv) == 2 and sys.argv[1] == "--help":
        return display_help()
    if len(sys.argv) == 1 or len(sys.argv) % 2 == 0:
        print("error args")
        return 84
    nb_arg = len(sys.argv)
    i = 1
    while i < nb_arg:
        if gen.add_config_nn(sys.argv[i], sys.argv[i + 1]) != 0:
            return 84
        i += 2
    gen.run()
    return 0

if __name__ == "__main__" :
    if main() == 84 :
        exit(84)
