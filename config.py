BOARD_SIZE = 19
CHANNEL_SIZE = 4

NUM_MOVES = 240

HIDDEN_SIZE = 128
BERT_LAYERS = 12
RES_CHANNELS = 128
RES_LAYERS = 14

FIRST_STEPS = ["dd", "cd", "dc", "dp", "dq", "cp", "pd", "qd", 
                   "pc", "pp", "pq", "qp","cc", "cq", "qc","qq"]

USE_MCTS = False
MCTS_BOUND = 10
MCTS_ITERS = 100
GAME_TYPE = "Combine"  # "Combine"(suggested), "Picture", "Word"