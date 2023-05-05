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
df = df[[x[0].get('type') == 'singleAnswer' for x in df['annotations']]]
questions = df["question"]
annotations = df["annotations"].apply(lambda x: x[0]["answer"][0])

# BPE encoding
t = Tokenizer()
encoded_corpus = t.train([questions, annotations], vocab_size)

q_toks = encoded_corpus[0]
a_toks = encoded_corpus[1]
q_data = [torch.tensor(tokens, dtype=torch.long) for tokens in q_toks]
a_data = [torch.tensor(tokens, dtype=torch.long) for tokens in a_toks]

# train/test split
q_percent = int(train_percent * len(q_data))
a_percent = int(train_percent * len(a_data))

q_train = q_data[:q_percent]
a_train = a_data[:a_percent]
q_test = q_data[:q_percent]
a_test = a_data[:a_percent]

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
    questions = q_train if split == 'train' else q_test
    annotations = a_train if split == 'train' else a_test 
    index = torch.randint(0, len(questions) - 1, size=(1,)).item()
    x = questions[index]
    y = annotations[index]
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch("train")
print(t.decode(xb.tolist()))
print(t.decode(yb.tolist()))

model = Transformer(block_size, vocab_size, n_embd, n_layer, n_head, dropout)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#
# for iter in range(max_iters):
#
#     # every once in a while evaluate the loss on train and val sets
#     if iter % eval_interval == 0 or iter == max_iters - 1:
#         losses = estimate_loss()
#         print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#
#     # sample a batch of data
#     xb, yb = get_batch('train')
#
#     # evaluate the loss
#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#
# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(t.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
