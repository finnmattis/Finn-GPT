import re
import pandas as pd

class Tokenizer:
    def __init__(self):
        self.__trained = False
        self.__merges = []
        self.vocab = []

    def pretokenize(self, text):
        text = text.lower() # all lowercase
        text = re.sub(r'[^a-z0-9\s.?!]', '', text) # remove weird unicode
        text = re.sub(r'\s+', ' ', text) # collapse whitespace
        text = re.sub(r'([.?!])', r' \1 ', text) # add space around ".", "?", "!"
        text.rstrip()
        text = ["<start>"] + list(text) + ["<end>"]
        return text

    def __get_most_frequent_pair(self, tokens):
        pairs = {}
        for char1, char2 in zip(tokens, tokens[1:]):
            pair = (char1, char2)
            pairs[pair] = pairs.get(pair, 0) + 1

        return max(pairs, key=pairs.get) if pairs else None

    def __tokens_to_nums(self, tokens):
        token_nums = []
        for token in tokens:
            token_nums.append(self.vocab.index(token))
        return token_nums

    def train(self, texts, vocab_size=1000):
        if self.__trained:
            raise RuntimeError("Already trained!")

        tokens_list = [self.pretokenize(text) for text in texts]
        tokens = [token for tokens_sublist in tokens_list for token in tokens_sublist]
        self.vocab = list(set([token for token in tokens]))
        
        # merge most common pair for vocab num_merges times
        for _ in range(vocab_size - len(self.vocab)):
            pair = self.__get_most_frequent_pair(tokens)
            if not pair:
                break

            token_new = "".join(pair)
            self.__merges.append(pair)
            
            updated_tokens = []
            skip_next = False
            # replace pair with new char
            for token1, token2 in zip(tokens, tokens[1:]):
                if skip_next:
                    skip_next = False
                    continue

                if token1 == pair[0] and token2 == pair[1]:
                    updated_tokens.append(token_new)
                    skip_next = True
                else:
                    updated_tokens.append(token1)

            if not skip_next:
                updated_tokens.append(tokens[-1])
            tokens = updated_tokens

            # add to vocab
            self.vocab.append(token_new)

        self.vocab = sorted(self.vocab)
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

        return self.__tokens_to_nums(tokens)

    def decode(self, encoded_text):
        decoded = []
        for token in encoded_text:
            decoded.append(self.vocab[token])
        return "".join(decoded)
