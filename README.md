# Go with ResNet and BERT: A Smart Go Engine Fusing Visual and Sequential Features

## 📌 Project Origin & Background (Capstone Project & Background)

This project originated as my **undergraduate capstone project** (Original repository: [BERT-for-GO-prediction](https://github.com/Vincent5201/BERT-for-GO-prediction)).

Traditional Go AI engines (such as AlphaGo and AlphaZero) rely heavily on Convolutional Neural Networks (CNN/ResNet) to process the board state as a **2D visual image** for feature extraction. In my capstone project, I explored an innovative, exploratory direction: **treating the move sequence in a Go game as a sequence of "words" in Natural Language Processing (NLP)**. By introducing the **BERT (Transformer)** model from NLP, I set out to validate the feasibility of predicting move placements using a pure text-sequence language model, as well as conducting a comparative analysis between the two distinct architectures.

### 🚀 Extensions & Advancements

Building upon the completion of my capstone project, I undertook significant **architectural expansions and performance optimizations** on the original system. This repository represents the fully expanded version, featuring deep advancements in **model fusion, decision-making algorithms, C++ backend acceleration, and search engines**.

Demo Video: [Demo Video (Google Drive)](https://drive.google.com/file/d/1e5euwbUa360M1Dj63Mja_KH-WQIOObLF/view?usp=drive_link)

---

## 🛠 Technical Implementations

Moving beyond simple single-model comparisons, I personally designed and implemented the following core modules during the extended development phase:

### 1. Combine Network (Hybrid Architecture)

To integrate **ResNet's 2D spatial image perception** with **BERT's historical move sequence modeling**, I designed a **multimodal fusion network (Combine Network)**:

* **Weight Freezing**: During the fusion training phase, the weights of the pre-trained ResNet (`modelR`) and BERT (`modelB`) models are frozen (`requires_grad = False`) to preserve their feature extraction capabilities and accelerate convergence.
* **Distribution Concatenation & Fusion**: The 361-dimensional move probability outputs from ResNet and BERT are normalized via Softmax and concatenated into a 722-dimensional composite feature vector.
* **MLP Fusion Layer**: A custom two-layer Multi-Layer Perceptron (722 -> 512 -> 361) is trained end-to-end to balance the weights between "spatial visual features" and "temporal sequence features," predicting the optimal next move placement.

### 2. Voting Ensemble Engine

Implemented a multi-model voting decision mechanism (`vote_next_move`) to provide dynamic decision-making:

* The system can simultaneously load multiple trained instances of BERT, ResNet, and Combine fusion models.
* At runtime, all models execute parallel inference and output their respective Softmax probability distributions. These outputs are aggregated via weighted ensemble voting to generate the final move selection probabilities, significantly enhancing decision robustness in complex game states.

### 3. C++ Performance Acceleration & pybind11 Integration

Move rule validation and local feature computation in Go are extremely CPU-intensive. To overcome Python's performance bottlenecks in loops and recursive operations, I rewrote the core algorithm in **C++** and bound it into a Python module using **pybind11**:

* **Capturing & Board Rules (`channel_01`)**: Implemented high-performance stone capture detection using C++ `std::set`. When a move is played, the algorithm recursively searches the liberties of adjacent opponent stones, instantly executing captures and updating the board state when liberties reach zero.
* **Liberties Feature Extraction (`channel_3`)**: Deeply optimized recursive liberty-counting algorithms. It rapidly calculates the liberties of each stone group on the board (capped at 6 liberties) and writes them directly into Channel 4 of the input feature matrix.
* **Bouzy's 5/21 Territory Evaluation Algorithm (`value_situation`)**: Implemented **Bouzy's 5/21 territory valuation algorithm** (5 dilations and 5 erosions, originally named 5/21). Utilizing high-speed C++ matrix operations to simulate the expansion and erosion of influence for both players, this serves as a high-precision, real-time territory evaluator and an alternative to Value Networks.

### 4. Custom Monte Carlo Tree Search (Custom MCTS)

Designed and implemented an MCTS engine tightly integrated with neural network policy outputs:

* **Selection**: Uses the Upper Confidence Bound (UCB) formula for node selection to balance exploration and exploitation.
* **Guided Expansion**: Instead of random expansion, child nodes are expanded using the top-$N$ (`nch=3` by default) high-probability candidate moves calculated by the convolutional or fusion network, drastically narrowing the search width.
* **Fast Simulation (Rollout)**: Employs fast-inference Policy Networks to guide move simulations up to a cap (`num_moves=240`), avoiding purely random rollouts.
* **Evaluation**: Upon reaching a terminal simulation state, the system invokes the C++-optimized Bouzy's 5/21 algorithm to evaluate territorial ownership, determining the simulation winner and backpropagating the result to update UCB weights.
* **Minimax Decision Making (`find_move`)**: Supports Minimax logic to dynamically maximize winning probability for the current player or minimize opponent win probability based on who is next to play.

### 5. Interactive Pygame Interface & Go Engine

Built a fully featured interactive visual Go game interface using Pygame:

* Supports **real-time rendering of black/white game states, precision mouse-click move placement, legal move checking, and move history undo (undo two moves / rollback)**.
* Offers multiple game and testing modes, including Human vs. AI and AI Self-Play.

---

## 📂 Project Architecture

* `config.py`: Global configuration file. Sets board dimensions (19x19), feature dimensions, model architecture hyperparameters (layers, hidden dimensions), and MCTS search iterations.
* `tools.py`: Helper functions. Includes move validity checks, conversion between SGF coordinates and 1D indices (e.g., `dp` -> 360), and top-$K$ accuracy (`myaccn`) evaluations.
* `gen_board.py`: Board feature generator. Dynamically restores move sequences into a 4-channel feature tensor (Channel 0: Black stones, Channel 1: White stones, Channel 2: Current player turn, Channel 3: Liberties per stone group).
* `mydatasets.py`: PyTorch Dataset implementation. Wraps `ResNetDataset`, `BERTDataset`, and `CombineDataset`, supporting dynamic generation of attention masks and token types.
* `models.py`: Neural network definitions. Contains PyTorch implementations of `myBert` (BERT Policy), `myResNet` (ResNet Policy), and `Combine` (Fusion Network).
* `train.py`: Model training script. Supports one-click training across various data sources and model architectures, with built-in CrossEntropyLoss and multi-class accuracy monitoring.
* `application.py`: Inference and prediction interface. Defines functions for single-step prediction, dual-feature prediction, and multi-model ensemble voting (`Vote Engine`).
* `score.py`: Evaluation module. Computes Accuracy, Top-5 Accuracy, Top-10 Accuracy, and F1-Scores (Micro & Macro) on test datasets.
* `mcts.py`: Monte Carlo Tree Search implementation. Defines MCTS tree nodes, selection and expansion, fast rollout, C++ territory evaluations, and weight backpropagation.
* `game.py`: Main Pygame Go application. Provides visual gameplay, candidate move visualization, and move-undo functionality.
* `cpp_setup.py` / `Makefile` / `cpptools.cpp`: C++ low-level optimization suit. Uses pybind11 to wrap C++ routines, significantly reducing computation latency for rule checking and territory evaluation.

---

## 📊 Model Input & Data Specifications

The system utilizes a **4-channel matrix feature representation** to model the 2D Go board state (`CHANNEL_SIZE = 4`):

| Channel Index | Feature Meanings | Value Range |
| --- | --- | --- |
| **Channel 0** | Black Stone Positions | 0 or 1 |
| **Channel 1** | White Stone Positions | 0 or 1 |
| **Channel 2** | Current Player Turn | All 0s (White) or All 1s (Black) |
| **Channel 3** | Group Liberties Feature | 0 to 6 (Capped at 6 liberties) |

* **BERT Sequence Input**: Maps 361 grid coordinates on the Go board directly to 361 token classes (plus padding token 362 and special token 363), with a default sequence length of `NUM_MOVES = 240`.
* **Attention Masking**: Dynamically masks padding positions during BERT training to ensure the model focuses strictly on valid move sequences.

---

## 🚀 How to Run

### 1. Build C++ Acceleration Core

Before running the project, compile the underlying C++ extension module.

**Using Python Setup (Recommended):**

```bash
python cpp_setup.py build_ext --inplace

```

*This generates a dynamic library file named `cpptools.cp3x-win32.pyd` (Windows) or `cpptools.so` (Linux/macOS) in the root directory, allowing direct import in Python.*

### 2. Launch GUI Interface

Once the C++ core is compiled successfully, run:

```bash
python game.py

```

* You can configure `GAME_TYPE` in `config.py` to `"Combine"`, `"Picture"`, or `"Word"` to toggle between different AI opponent architectures.
* Set `USE_MCTS = True` to enable MCTS search enhancement for AI move selections.

### 3. Training and Evaluation

**Train Neural Networks:**

```bash
python train.py

```

* Modify `data_config` and `model_config` in `train.py` to train BERT, ResNet, or Combine models.

**Evaluate Model Metrics:**

```bash
python score.py

```

* Loads the designated test set (e.g., Foxwq 9d professional game records) to compute Top-$K$ prediction accuracies and F1-Scores.
