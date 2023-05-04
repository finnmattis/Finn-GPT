import torch
import torch.nn as nn
from torch.nn import functional as F
import pandas as pd

from tokens import Tokenizer
from transformer import Transformer

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 100
max_iters = 100
eval_interval = 10
learning_rate = 1e-3
eval_iters = 200
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
tokens = t.train(questions, vocab_size)
data = torch.tensor(tokens, dtype=torch.long)

# train/test split
n = int(train_percent*len(data))
train_data = data[:n]
test_data = data[n:]

# estimate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else train_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

model = Transformer(block_size, vocab_size, n_embd, n_layer, n_head, dropout)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')


# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(t.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
