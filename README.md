# Go-on-Bert-and-ResNet
## Introducion
* Reconsturct codes in my [old repo](https://github.com/Vincent5201/BERT-for-GO-prediction) and remove redundant parts.
* A simple Go engine refers to AlphaGo.
* For the policy network, besides the traditional ResNet, we also tried using BERT.
* For the value network, use a computational method instead of deep learning model, it is useless now.
* Implement MCTS functoins to predict next move, but we don't have usefull value network.
* No Reinforce learning.
* Add pybind11 and use c++ to improve running speed.
