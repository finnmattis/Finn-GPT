import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionHead(nn.Module):
    def __init__(self, block_size, head_size, n_embd, dropout, mask=False):
        super().__init__()
        self.mask = mask
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T) scaling makes gradients smoother
        if self.mask:
            # masking only done on decoders so that tokens don't have access to future tokens
            wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class CrossAttentionHead(nn.Module):
    def __init__(self, block_size, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    # keys and values come from encoder, queries come from decoder
    def forward(self, x, y):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        k = self.key(x)  # (B,T,hs)
        q = self.query(y)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T) scaling makes gradients smoother
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, block_size, num_heads, head_size, n_embd, dropout, decoder=False
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                SelfAttentionHead(block_size, head_size, n_embd, dropout, decoder)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, block_size, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CrossAttentionHead(block_size, head_size, n_embd, dropout)
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        out = torch.cat([h(x, y) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, block_size, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(
            block_size, n_head, head_size, n_embd, dropout, decoder=False
        )
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, block_size, n_embd, n_head, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadSelfAttention(
            block_size, n_head, head_size, n_embd, dropout, decoder=True
        )
        self.ca = MultiHeadCrossAttention(
            block_size, n_head, head_size, n_embd, dropout
        )
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.ln4 = nn.LayerNorm(n_embd)

    def forward(self, enc_out, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ca(self.ln2(enc_out), self.ln3(x))
        x = x + self.ffwd(self.ln4(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        block_size,
        vocab_size,
        n_embd,
        n_layer,
        n_head,
        dropout,
        device,
    ):
        super().__init__()
        self.block_size = block_size  # needed for generate method
        self.device = device
        # each token directly reads off the logits for the next token from a lookup table
        self.encoder_token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.decoder_token_embeddings = nn.Embedding(vocab_size, n_embd)
        self.encoder_position_embeddings = nn.Embedding(block_size, n_embd)
        self.decoder_position_embeddings = nn.Embedding(block_size, n_embd)
        self.encoder_blocks = nn.Sequential(
            *[EncoderBlock(block_size, n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.decoder_blocks = [
            DecoderBlock(block_size, n_embd, n_head, dropout) for _ in range(n_layer)
        ]
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B, T = idx.shape

        # encoder side:
        encoder_tok_emb = self.encoder_token_embeddings(idx)  # (B,T,C)
        encoder_pos_emb = self.encoder_position_embeddings(
            torch.arange(T, device=self.device)
        )
        enc_out = encoder_tok_emb + encoder_pos_emb  # (B,T,C)
        enc_out = self.encoder_blocks(enc_out)

        # decoder side (autoregressive):
        B, T, C = enc_out.shape
        decoder_context = torch.zeros((B, 1), dtype=torch.long, device=self.device)
        loss = 0
        num_tokens_to_predict = targets.shape[1] if targets is not None else 20

        for i in range(num_tokens_to_predict):
            decoder_tok_emb = self.decoder_token_embeddings(
                decoder_context
            )  # (B, T, C)
            decoder_pos_emb = self.decoder_position_embeddings(
                torch.arange(decoder_context.shape[1], device=self.device)
            )  # (T, C)
            dec_emb = decoder_tok_emb + decoder_pos_emb  # (B, T, C)

            dec_block_out = dec_emb
            for decoder_block in self.decoder_blocks:
                dec_block_out = decoder_block(enc_out, dec_block_out)  # (B, T, C)
            dec_ln_out = self.ln_f(dec_block_out)  # (B, T, C)
            logits = self.lm_head(dec_ln_out)  # (B, T, C)

            # Compute the loss for the current token
            if targets is not None:
                curr_loss = F.cross_entropy(logits[:, -1, :], targets[:, i])
                loss += curr_loss

            # Update the decoder context
            next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            decoder_context = torch.cat([decoder_context, next_token], dim=1)
            # i get bored easily:
            if targets is not None:
                print(i, targets.shape[1])

        # Divide the accumulated loss by the number of tokens to get the average loss per token
        loss /= num_tokens_to_predict
        return decoder_context, loss
