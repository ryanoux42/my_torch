# ♟️ MY_TORCH

## Overview

**MY_TORCH** is a **neural network built from scratch** in **Python** (only **NumPy**, no PyTorch / TensorFlow) that analyses a **chessboard** and predicts its state.

Each board is given in **[FEN notation](https://en.wikipedia.org/wiki/Forsyth–Edwards_Notation)** and the network predicts whether the position is **`Check`** or **`Nothing`**.

The project is split into two programs:
- **`my_torch_generator`** → creates new, randomly initialised networks from a JSON configuration
- **`my_torch_analyzer`** → trains a network on a dataset or uses it to make predictions

This project focuses on **forward / backward propagation**, **activation functions**, **optimisers**, **regularisation** and **hyperparameter tuning**.

---

## 🧠 Network Architecture

| Layer | Size | Activation |
|-------|------|------------|
| Input | 64 (one value per square) | — |
| Hidden 1 | 128 | Leaky ReLU + Dropout |
| Hidden 2 | 64 | Leaky ReLU + Dropout |
| Hidden 3 | 32 | Leaky ReLU + Dropout |
| Output | 1 | Sigmoid |

### Board encoding

Each of the 64 squares is converted into a number:

| Piece | White | Black |
|-------|-------|-------|
| Pawn | `1` | `-1` |
| Knight | `2` | `-2` |
| Bishop | `3` | `-3` |
| Rook | `4` | `-4` |
| Queen | `5` | `-5` |
| King | `6` | `-6` |
| Empty | `0` | `0` |

### Training features

- Mini-batch gradient descent
- **Adam** optimiser (`beta1`, `beta2`)
- **Dropout** to prevent overfitting
- **L2 regularisation** (`lambda`)
- Learning-rate decay (halved every 200 epochs)
- Networks saved / loaded with `pickle` (`.nn` files)

---

## ⚙️ Usage

### Compilation

```bash
make
```

This creates the two executables `my_torch_generator` and `my_torch_analyzer` at the root of the repository.

Other rules:
```bash
make re      # Rebuild everything
make clean   # Remove caches
make fclean  # Remove executables
```

### Prerequisites

- Python 3
- NumPy

---

### 1. Generate networks

```bash
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2 ...]
```

#### Example:
```bash
./my_torch_generator config_file.json 3
# -> config_file_1.nn, config_file_2.nn, config_file_3.nn
```

#### Configuration file

```json
{
  "input_nb": 64,
  "hidden_layer_1_nb": 128,
  "hidden_layer_2_nb": 64,
  "hidden_layer_3_nb": 32,
  "output_nb": 1,
  "learning_rate": 0.1,
  "dropout": 0.25,
  "beta1": 0.9,
  "beta2": 0.999,
  "lambda": 0.01,
  "epochs": 1000
}
```

---

### 2. Train a network

```bash
./my_torch_analyzer --train [--save SAVEFILE] LOADFILE CHESSFILE
```

#### Example:
```bash
./my_torch_analyzer --train --save trained.nn config_file_1.nn train.txt
```

### 3. Predict

```bash
./my_torch_analyzer --predict LOADFILE CHESSFILE
```

#### Example:
```bash
./my_torch_analyzer --predict trained.nn boards.txt
```

---

## 📄 Dataset Format

One board per line: the FEN string followed by the expected state.

```
8/8/4k3/7R/1P5P/2PK4/8/4r3 b - - 0 46 Nothing
2k5/8/3Kp3/6p1/1R4P1/4n3/8/8 b - - 4 55 Nothing
```

`train.txt` contains **40,000** labelled positions (20,000 `Check` / 20,000 `Nothing`).

---

## 📚 Resources

- [Learning to evaluate chess positions with deep neural networks (ICPRAM 2018)](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf)
- [Building a Multi-Layer Perceptron from scratch with NumPy](https://elcaiseri.medium.com/building-a-multi-layer-perceptron-from-scratch-with-numpy-e4cee82ab06d)
- [Overfitting](https://en.wikipedia.org/wiki/Overfitting) · [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)

---

## 🫂 Team

- Ryan Hercule
- Jean-Garice Denoncin
