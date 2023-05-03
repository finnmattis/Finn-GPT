import pandas as pd

def get_most_frequent_pair(tokens):
    pairs = {}
    for char1, char2 in zip(tokens, tokens[1:]):
        pair = (char1, char2)
        pairs[pair] = pairs.get(pair, 0) + 1

    return max(pairs, key=pairs.get) if pairs else None

def bpe(texts, num_merges=1000):
    tokens_list = [list(text) for text in texts]
    tokens = [token for tokens_sublist in tokens_list for token in tokens_sublist]
    vocab = {token: 1 for token in tokens}
    merges = []

    for _ in range(num_merges):
        pair = get_most_frequent_pair(tokens)
        if not pair:
            break

        token_new = "".join(pair)
        merges.append(pair)
        
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

    return tokens, vocab, merges

def encode(text, merges):
    tokens = list(text)

    for merge in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if i < len(tokens) - 1 and (token, tokens[i + 1]) in merges:
                new_tokens.append("".join((token, tokens[i + 1])))
                i += 2
            else:
                new_tokens.append(token)
                i += 1
        tokens = new_tokens

    return tokens



# text = "This is a sample text for BPE encoding."
# tokens, vocab, merges = bpe(text, num_merges=10)
# print("Encoded Tokens:", tokens)
# print("Vocab:", vocab)
#
# novel_text = "This is a sample text for BPE encoding."
# encoded_novel_text = encode(novel_text, merges)
# print("Encoded Novel Text:", encoded_novel_text)

df = pd.read_json("data.json")
tokens, vocab, merges = bpe(list(df["question"]), 100)
print(vocab)
print("\n\n\n")
print(merges)
