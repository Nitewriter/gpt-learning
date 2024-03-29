import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embedding_size % num_heads == 0
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

        self.fc_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, query, key=None, value=None):
        batch_size = query.shape[0]

        if key is None and value is None:
            key = query
            value = query

        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)

        # Split the embedding into num_heads and concatenate
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Calculate attention scores
        energy = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / (self.embedding_size**0.5)

        # Apply attention mask if needed
        # (Note: GPT-2 does not use attention mask as it operates on autoregressive models)

        attention = torch.softmax(energy, dim=-1)
        x = torch.einsum("nhql,nlhd->nqhd", [attention, V]).reshape(
            batch_size, -1, self.embedding_size
        )

        x = self.fc_out(x)
        return x
