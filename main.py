import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

from tokens import Tokenizer
from transformer import Transformer

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_merges = 100
train_percent = 0.9
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
n_embd = 200
n_head = 6
n_layer = 6
dropout = 0.2

# read from json
df = pd.read_json("data.json")
questions = df["question"]

# BPE encoding
t = Tokenizer()
tokens = t.train(questions, num_merges)
vocab_size = len(t.vocab)
data = torch.tensor(tokens, dtype=torch.long)

# train/test split
n = int(train_percent*len(data))
train_data = data[:n]
test_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = Transformer(block_size, vocab_size, n_embd, n_layer, n_head, dropout)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
