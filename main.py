import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from tokens import Tokenizer
from transformer import Transformer

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 100
max_iters = 100
eval_interval = 10
learning_rate = 1e-3
eval_iters = 10
train_percent = 0.9
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 150  # what is the maximum context length for predictions?
n_embd = 200
n_head = 6
n_layer = 6
dropout = 0.2
dec_start_state = torch.zeros((batch_size, 1), dtype=torch.long)

# read from json
df = pd.read_json("data.json")
df = df[[x[0].get("type") == "singleAnswer" for x in df["annotations"]]]
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


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    questions = q_train if split == "train" else q_test
    annotations = a_train if split == "train" else a_test

    ix = torch.randint(len(questions) - block_size, (batch_size,))
    x = [questions[i] for i in ix]
    y = [annotations[i] for i in ix]

    x = torch.stack(
        [
            F.pad(i, (0, max(0, block_size - len(i))), mode="constant", value=0)
            for i in x
        ]
    )
    y = torch.stack(
        [
            F.pad(i, (0, max(0, block_size - len(i))), mode="constant", value=0)
            for i in y
        ]
    )
    x, y = x.long(), y.long()
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch("train")

model = Transformer(
    block_size, vocab_size, n_embd, n_layer, n_head, dropout, device, dec_start_state
)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# # generate from the model
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(t.decode(m.generate(context, max_new_tokens=500)[0].tolist()))
