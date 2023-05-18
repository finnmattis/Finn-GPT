import json
import os
import re

import pandas as pd


class Tokenizer:
    def __init__(self):
        self.__trained = False
        self.__merges = []
        self.vocab = []

    def pretokenize(self, text):
        text = text.lower()  # all lowercase
        text = re.sub(r"[^a-z0-9\s.?!]", "", text)  # remove weird unicode
        text = re.sub(r"\s+", " ", text)  # collapse whitespace
        text = re.sub(r"([.?!])", r" \1 ", text)  # add space around ".", "?", "!"
        text.rstrip()
        text = ["<start>"] + list(text) + ["<end>"]
        return text

    def __get_most_frequent_pair(self, tokens):
        pairs = {}
        for block in tokens:
            for example in block:
                for tok1, tok2 in zip(example, example[1:]):
                    # Don't want any pairs with start b.c. start is provided by me - the model should not be able to generate start in the middle
                    if tok1 == "<start>":
                        continue
                    pair = (tok1, tok2)
                    pairs[pair] = pairs.get(pair, 0) + 1
        return max(pairs, key=pairs.get) if pairs else None

    def __tokens_to_nums(self, tokens):
        token_nums = []
        for block in tokens:
            block_nums = []
            for example in block:
                example_nums = []
                for token in example:
                    example_nums.append(self.vocab.index(token))
                block_nums.append(example_nums)
            token_nums.append(block_nums)
        return token_nums

    # input: multi-dimensional array of examples (n-dimensional array of strings)
    def train(self, corpus, vocab_size=1000):
        if self.__trained:
            raise RuntimeError("Already trained!")

        # want to keep blocks seperate
        tokens = [[self.pretokenize(example) for example in block] for block in corpus]
        self.vocab = list(
            set(char for block in tokens for example in block for char in example)
        )

        # merge most common pair for vocab num_merges times
        for _ in range(vocab_size - len(self.vocab)):
            pair = self.__get_most_frequent_pair(tokens)
            if not pair:
                break

            token_new = "".join(pair)
            self.__merges.append(pair)

            # replace seperated tokens with new_merged token for each block of "tokens"
            updated_tokens = []
            for block in tokens:
                updated_block = []
                for example in block:
                    skip_next = False
                    updated_example = []
                    for tok1, tok2 in zip(example, example[1:]):
                        if skip_next:
                            skip_next = False
                            continue
                        if tok1 == pair[0] and tok2 == pair[1]:
                            updated_example.append(token_new)
                            skip_next = True
                        else:
                            updated_example.append(tok1)
                    updated_block.append(updated_example)
                updated_tokens.append(updated_block)
            # update tokens and vocab
            tokens = updated_tokens
            self.vocab.append(token_new)

        self.vocab = sorted(self.vocab)
        self.vocab.insert(0, self.vocab.pop(self.vocab.index("<start>")))
        self.vocab.insert(1, self.vocab.pop(self.vocab.index("<end>")))

        self.__trained = True
        return self.__tokens_to_nums(tokens)

    def encode(self, text):
        tokens = self.pretokenize(text)

        for merge in self.__merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if i < len(tokens) - 1 and (token, tokens[i + 1]) in self.__merges:
                    new_tokens.append("".join((token, tokens[i + 1])))
                    i += 2
                else:
                    new_tokens.append(token)
                    i += 1
            tokens = new_tokens
        # a bunch of weird list stuff needed because of self.train()
        return self.__tokens_to_nums([[tokens]])[0][0]

    def decode(self, encoded_text):
        if encoded_text[0] != 0:
            raise RuntimeError('First token is not "<start>"')

        decoded = []
        for token in encoded_text[1:-1]:
            decoded.append(self.vocab[token])

        return "".join(decoded)

    def save_state(self, path):
        with open(path, "w") as file:
            json.dump({"vocab": self.vocab, "merges": self.__merges}, file)

    def load_state(self, path):
        with open(path, "r") as file:
            data = json.load(file)
            if not len(data) == 2 or "vocab" not in data or "merges" not in data:
                raise RuntimeError(
                    'Data not foratted properly! Should have "vocab" and "merges"'
                )
            self.vocab = data["vocab"]
            self.__merges = data["merges"]
            self.__trained = True
