import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from hyperparameters import HYPERPARAMETERS as HP
from tokens import Tokenizer
from transformer import Transformer

# read from json
df = pd.read_json("data.json")
df = df[[x[0].get("type") == "singleAnswer" for x in df["annotations"]]]
questions = df["question"]
annotations = df["annotations"].apply(lambda x: x[0]["answer"][0])

# BPE encoding
t = Tokenizer()
encoded_corpus = t.train([questions, annotations], HP["vocab_size"])
t.save_state("artifacts/bpe.json")

q_toks = encoded_corpus[0]
a_toks = encoded_corpus[1]
q_data = [torch.tensor(tokens, dtype=torch.long) for tokens in q_toks]
a_data = [torch.tensor(tokens, dtype=torch.long) for tokens in a_toks]

# train/test split
q_percent = int(HP["train_percent"] * len(q_data))
a_percent = int(HP["train_percent"] * len(a_data))

q_train = q_data[:q_percent]
a_train = a_data[:a_percent]
q_test = q_data[q_percent:]
a_test = a_data[a_percent:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    questions = q_train if split == "train" else q_test
    annotations = a_train if split == "train" else a_test

    ix = torch.randint(len(questions) - HP["block_size"], (HP["batch_size"],))
    x = [questions[i] for i in ix]
    y = [annotations[i] for i in ix]

    x = torch.stack(
        [
            F.pad(i, (0, max(0, HP["block_size"] - len(i))), mode="constant", value=0)
            for i in x
        ]
    )
    y = torch.stack(
        [
            F.pad(i, (0, max(0, HP["block_size"] - len(i))), mode="constant", value=0)
            for i in y
        ]
    )
    x, y = x.long(), y.long()
    x, y = x.to(HP["device"]), y.to(HP["device"])
    return x, y


model = Transformer(
    HP["block_size"],
    HP["vocab_size"],
    HP["n_embd"],
    HP["n_layer"],
    HP["n_head"],
    HP["dropout"],
    HP["device"],
)
m = model.to(HP["device"])
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=HP["learning_rate"])

for i in range(HP["max_iters"]):
    print(f"Batch: {i+1}/{HP['max_iters']}")
    # sample a batch of data
    xb, yb = get_batch("train")
    # evaluate the loss
    logits, loss = model(xb, yb)
    print(f"Loss: {loss}")
    # take gradient step
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()

# save state
torch.save(model.state_dict(), "artifacts/checkpoint.pth")
# generate from the model
