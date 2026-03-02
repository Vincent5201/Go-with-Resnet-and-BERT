# Go with ResNet and BERT

## Overview
This project was originally developed as my undergraduate thesis (see the [old repo](https://github.com/Vincent5201/BERT-for-GO-prediction)).

The core idea was to treat the game of Go as a sequence of text tokens and use BERT as the prediction model. Traditionally, Go AI models rely on image-based architectures such as ResNet, but this project explored the feasibility of using a text-based Transformer model instead. The original repository contains detailed comparisons and analysis.

After the thesis was completed, I continued expanding the project (detailed in next paragraph). This project serves both as a learning experience and as a playable Go engine that allows direct matches against trained models.

> Key words: Go Engine Design, PyTorch, ResNet, BERT, Monte Carlo Tree Search (MCTS), Bouzy’s 5/21 Algorithm, pybind11, Python–C++ Integration.

> demo: https://drive.google.com/file/d/1e5euwbUa360M1Dj63Mja_KH-WQIOObLF/view?usp=drive_link

## Key Features
### Engine Architecture
* A simplified Go engine design inspired by AlphaGo.
* Monte Carlo Tree Search (MCTS) for strategic move selection.
* No reinforcement learning used (purely supervised training + Bouzy’s 5/21 Algorithm).
* C++ optimizations for performance-critical calculations (Python–C++ integration via pybind11).
* Pygame-based interactive Go interface.

### Neural Network Models
* Policy Network
    * Traditional ResNet-based architecture.
    * BERT-based architecture (experimental approach treating Go as a text sequence).
* Value Network
    * Board evaluation based on Bouzy’s 5/21 Algorithm instead of neural network model.

> Note: The current BERT model weights appear to be corrupted.



## File Descriptions

- `config.py`: Configuration constants including board size, model hyperparameters, and game settings
- `tools.py`: Utility functions for move conversion, validation, and accuracy calculations
- `gen_board.py`: Board state generation and sequence processing functions
- `mydatasets.py`: PyTorch dataset classes for different model types (BERT, ResNet, Combine)
- `models.py`: Neural network model definitions (myBert, myResNet, Combine)
- `train.py`: Training script for model optimization
- `application.py`: Core application functions for prediction and move selection
- `score.py`: Evaluation and scoring utilities
- `mcts.py`: Monte Carlo Tree Search implementation for strategic play
- `game.py`: Pygame-based interactive Go game with AI opponent
- `cpp_setup.py`: Setup script for C++ extensions
- `cpptools.cpp`: C++ functions for board rule processing (captures, liberties)
- `Makefile`: Build configuration for C++ components

## How to run
1. run `python cpp_setup.py build_ext --inplace`
2. play: `python game.py` or train: `python train.py`
