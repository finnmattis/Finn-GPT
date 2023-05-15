import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VOCAB_SIZE = 100
MAX_ITERS = 25
EVAL_INTERVAL = 10
LEARNING_RATE = 1e-3
EVAL_ITERS = 10
TRAIN_PERCENT = 0.9
BATCH_SIZE = 32  # how many independent sequences will we process in parallel?
BLOCK_SIZE = 115  # what is the maximum context length for predictions?
N_EMBD = 200
N_HEAD = 6
N_LAYER = 2
DROPOUT = 0.2

HYPERPARAMETERS = {
    "device": DEVICE,
    "vocab_size": VOCAB_SIZE,
    "max_iters": MAX_ITERS,
    "eval_interval": EVAL_INTERVAL,
    "learning_rate": LEARNING_RATE,
    "eval_iters": EVAL_ITERS,
    "train_percent": TRAIN_PERCENT,
    "batch_size": BATCH_SIZE,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "dropout": DROPOUT,
}
