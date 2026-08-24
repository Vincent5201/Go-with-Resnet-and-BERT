# Go with ResNet and BERT

## Overview

### Background

This project originated as my undergraduate capstone project ([Original Repo](https://github.com/Vincent5201/BERT-for-GO-prediction)).

* Traditional Approach: Treats the Go board as a 2D visual image, using convolutional neural networks (CNN/ResNet) for feature extraction of board states (e.g., AlphaGo, AlphaZero, Katago).
* Capstone Innovation: Replaces the visual paradigm by treating the move sequence of a Go game as a sequence of "words" in Natural Language Processing (NLP).
  * Each of the 361 board positions is mapped directly to a unique "word" in the NLP vocabulary.
  * Leverages the BERT model to validate whether a pure text-sequence language model can accurately predict move placements.
* Evaluates and compares performance across two fundamentally distinct architectures: image-based convolution (CNN) vs. sequence-based language modeling (BERT).

### Extensions & Advancements
Building upon the completion of my capstone project, I undertook expansions and optimizations on the original system.
* C++ Engine Optimization: Rewrote core game logic, liberty tracking, and territory estimation in C++ using pybind11 to eliminate Python performance bottlenecks.
* Custom Policy-Guided MCTS: Implemented a MCTS algorithm guided by neural network predictions and C++ evaluation functions to improve search efficiency.
* Interactive Pygame GUI: Built a real-time graphical interface supporting board visualization, move history rollback, and Human vs. AI / AI Self-Play modes.

> Key words: Go Engine Design, PyTorch, ResNet, BERT, Monte Carlo Tree Search (MCTS), Bouzy’s 5/21 Algorithm, pybind11, Python–C++ Integration.


Demo Video: [Google Drive Link](https://drive.google.com/file/d/1e5euwbUa360M1Dj63Mja_KH-WQIOObLF/view?usp=drive_link)

https://github.com/user-attachments/assets/31873403-3bc8-4e22-97e7-e72877d94648



---

## 🛠 Key Implementations

### 1. Dual-Model Benchmark & Ensemble
> undergraduate capstone project part

* Behavioral Comparison: Analyzes tactical and strategic differences in Go playing styles between visual and sequence representations.
* Hybrid Fusion: Merges frozen pre-trained ResNet and BERT features into a unified model to leverage both strengths.
* Voting Ensemble: Aggregates predictions across multiple models via weighted voting for superior decision robustness.

### 2. C++ Optimization & pybind11

To resolve Python performance bottlenecks in loops and recursion, core game logic was rewritten in **C++** and bound via **pybind11**:

* **Rules & Capturing (`channel_01`)**: Fast recursive liberty tracking and stone capturing using `std::set`.
* **Liberty Feature (`channel_3`)**: Optimized recursive algorithm tracking liberties per stone group (capped at 6).
* **Territory Evaluation (`value_situation`)**: C++ implementation of **Bouzy's 5/21 algorithm** (5 dilations/erosions) for real-time territory estimation.

### 3. Custom Policy-Guided MCTS

* **Selection**: Uses standard UCB formula to balance exploration and exploitation.
* **Guided Expansion & Rollout**: Expands only top-$N$ candidate moves (`nch=3`) predicted by policy networks; rollouts are policy-guided up to a depth limit (`num_moves=240`).
* **Evaluation**: Terminal states are evaluated directly via C++ Bouzy's algorithm to update UCB weights.

### 4. Interactive Pygame GUI

Features real-time board rendering, legal move checks, move history rollback, and multiple game modes (Human vs. AI / AI Self-Play).

---

## 📂 Project Architecture

* `config.py`: Global configurations (board size, hyperparameters, MCTS parameters).
* `tools.py` / `gen_board.py`: SGF parsing, coordinate conversions, and 4-channel tensor generation.
* `mydatasets.py` / `models.py`: PyTorch datasets and architectures (`myBert`, `myResNet`, `Combine`).
* `train.py` / `score.py`: Training pipeline and metric evaluation (Top-$K$ Accuracy, F1-Score).
* `mcts.py` / `application.py`: MCTS implementation, single-step inference, and voting engine.
* `game.py`: Pygame-based UI application.
* `cpptools.cpp` / `cpp_setup.py`: C++ source code and pybind11 setup module.

---

## 📊 Data Specifications

The engine represents 2D board states using a **4-channel matrix** (`CHANNEL_SIZE = 4`):

| Channel | Description | Value |
| --- | --- | --- |
| **Channel 0** | Black Stone Positions | 0 or 1 |
| **Channel 1** | White Stone Positions | 0 or 1 |
| **Channel 2** | Current Player Turn | 0 (White) / 1 (Black) |
| **Channel 3** | Group Liberties Feature | 0 to 6 |

* **BERT Input**: Maps 361 board coordinates to sequence tokens (length `NUM_MOVES = 240`) with dynamic attention masking.

---

## 🚀 Getting Started

### 1. Build C++ Extension

```bash
python cpp_setup.py build_ext --inplace

```

### 2. Launch Interface

```bash
python game.py

```

*(Toggle `GAME_TYPE` or enable `USE_MCTS = True` in `config.py`)*

### 3. Train & Evaluate

```bash
python train.py  # Train models
python score.py  # Run benchmark tests

```
