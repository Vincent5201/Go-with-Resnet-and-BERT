# Go with ResNet and BERT

## Overview
This project explores an innovative approach to Go AI by shifting from traditional visual recognition to an NLP sequence-modeling paradigm, alongside C++ engine optimizations and an interactive interface.

* **Capstone Core**: Replaces traditional 2D image processing (CNN/ResNet) by treating Go move sequences as "words" in NLP. Uses **BERT** to validate sequence-based move prediction and benchmarks performance against visual architectures.
* **Engine & MCTS Extensions**: Accelerates core game logic, liberty tracking, and Bouzy's territory estimation via **C++ (pybind11)**, paired with a custom policy-guided **MCTS** engine.
* **Interactive Play**: Includes a real-time **Pygame GUI** supporting Human vs. AI and AI Self-Play modes.

> **Keywords**: Go Engine Design, PyTorch, ResNet, BERT, Monte Carlo Tree Search (MCTS), Bouzy’s 5/21 Algorithm, pybind11, Python–C++ Integration.

Demo Video: [Google Drive Link](https://drive.google.com/file/d/1e5euwbUa360M1Dj63Mja_KH-WQIOObLF/view?usp=drive_link)

https://github.com/user-attachments/assets/31873403-3bc8-4e22-97e7-e72877d94648


---

## 🛠 Key Implementations
### 1. Dual-Model Benchmark & Ensemble
*(Undergraduate Capstone Part)*
* **Behavioral Comparison**: Analyzes tactical and strategic differences in playing styles between visual (ResNet) and sequence (BERT) representations.
* **Hybrid Fusion**: Merges frozen pre-trained ResNet and BERT features into a unified backbone to leverage both spatial and sequential strengths.
* **Voting Ensemble**: Aggregates predictions across multiple model architectures via weighted voting for enhanced decision robustness.

### 2. C++ Optimization & pybind11
Rewrote performance-critical loops and recursive functions in **C++** bound via **pybind11**:
* **Rules & Capturing (`channel_01`)**: Fast recursive liberty tracking and stone capturing using `std::set`.
* **Liberty Feature (`channel_3`)**: Optimized recursive tracking of group liberties (capped at 6).
* **Territory Evaluation (`value_situation`)**: C++ implementation of **Bouzy's 5/21 algorithm** (5 dilations/erosions) for real-time territory estimation.

### 3. Custom Policy-Guided MCTS
* **Selection**: Uses the standard UCB formula to balance exploration and exploitation.
* **Guided Expansion & Rollout**: Expands top-$N$ candidate moves (`nch=3`) predicted by policy networks; rollouts are policy-guided up to depth limit (`num_moves=240`).
* **Evaluation**: Evaluates terminal states directly via C++ Bouzy's algorithm to update UCB weights.

### 4. Interactive Pygame GUI
A real-time graphical interface featuring legal move validation, move history rollback, and customizable play modes (Human vs. AI / AI Self-Play).
---

## 📂 Project Architecture

* `config.py`: Global configurations (board size, hyperparameters, MCTS settings).
* `tools.py` / `gen_board.py`: SGF parsing, coordinate conversions, and 4-channel tensor generation.
* `mydatasets.py` / `models.py`: PyTorch datasets and model architectures (`myBert`, `myResNet`, `Combine`).
* `train.py` / `score.py`: Training pipeline and benchmark evaluation (Top-$K$ Accuracy, F1-Score).
* `mcts.py` / `application.py`: MCTS implementation, inference logic, and voting engine.
* `game.py`: Pygame-based UI application.
* `cpptools.cpp` / `cpp_setup.py`: Core C++ logic and pybind11 compilation setup.
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
