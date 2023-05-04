import pandas as pd

class Tokenizer:
    def __init__(self):
        self.merges = []

    def __get_most_frequent_pair(self, tokens):
        pairs = {}
        for char1, char2 in zip(tokens, tokens[1:]):
            pair = (char1, char2)
            pairs[pair] = pairs.get(pair, 0) + 1

        return max(pairs, key=pairs.get) if pairs else None

    def train(self, texts, num_merges=1000):
        tokens_list = [["<start>"] + list(text) + ["<end>"] for text in texts]
        tokens = [token for tokens_sublist in tokens_list for token in tokens_sublist]
        vocab = {token: 1 for token in tokens}

        for _ in range(num_merges):
            pair = self.__get_most_frequent_pair(tokens)
            if not pair:
                break

            token_new = "".join(pair)
            self.merges.append(pair)
            
            updated_tokens = []
            skip_next = False
            for i in range(len(tokens) - 1):
                if skip_next:
                    skip_next = False
                    continue

                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    updated_tokens.append(token_new)
                    skip_next = True
                else:
                    updated_tokens.append(tokens[i])

            if not skip_next:
                updated_tokens.append(tokens[-1])

            tokens = updated_tokens
            vocab[token_new] = 1
        return len(vocab)

    def encode(self, text):
        tokens = list(text)

        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if i < len(tokens) - 1 and (token, tokens[i + 1]) in self.merges:
                    new_tokens.append("".join((token, tokens[i + 1])))
                    i += 2
                else:
                    new_tokens.append(token)
                    i += 1
            tokens = new_tokens

        return tokens

mytok = Tokenizer()

df = pd.read_json("data.json")
mytok.train(list(df["question"]), 100)

novel_text = "Who died today?"
print(mytok.encode(novel_text))
