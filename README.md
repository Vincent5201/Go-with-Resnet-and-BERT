# Go-with-Resnet-and-BERT

## Introducion
* Reconsturct codes in my [old repo](https://github.com/Vincent5201/BERT-for-GO-prediction) and remove redundant parts.
* A simple Go engine refers to AlphaGo.
* For the policy network, besides the traditional ResNet, we also tried using BERT.
* For the value network, use Bouzy's 5/21 Algorithm.
* Implement MCTS algorithm to predict next move.
* No Reinforce learning.
* Use pybind11 and c++ to improve running speed.
* Weight of BERT seems to be broken.

## How to run
1. run `python cpp_setup.py build_ext --inplace`
2. play: `python game.py` or train: `python train.py`
