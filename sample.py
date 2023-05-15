import torch
from flask import Flask, jsonify, request
from torch.nn import functional as F

from hyperparameters import HYPERPARAMETERS as HP
from tokens import Tokenizer
from transformer import Transformer

t = Tokenizer()
t.load_state("artifacts/bpe.json")

model = Transformer(
    HP["block_size"],
    HP["vocab_size"],
    HP["n_embd"],
    HP["n_layer"],
    HP["n_head"],
    HP["dropout"],
    HP["device"],
)
model.load_state_dict(torch.load("artifacts/checkpoint.pth"))

app = Flask(__name__)


@app.route("/response/<question>", methods=["GET"])
def get_data(question):
    return jsonify({"response": get_reponse(question)})


def get_reponse(question):
    text = torch.tensor(t.encode(question), dtype=torch.long)
    text = F.pad(
        text, (0, max(0, HP["block_size"] - len(text))), mode="constant", value=0
    ).unsqueeze(0)
    logits, _ = model(text)
    print(f"{question}: {t.decode(logits.tolist()[0])}")
    return t.decode(logits.tolist()[0])


if __name__ == "__main__":
    app.run()
