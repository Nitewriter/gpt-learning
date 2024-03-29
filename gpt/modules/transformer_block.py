import torch
import torch.nn as nn

from .multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, hidden_size, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embedding_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_size),
        )
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.attention(x)
        x = x + self.dropout(self.ln1(attention_output))
        feed_forward_output = self.feed_forward(x)
        x = x + self.dropout(self.ln2(feed_forward_output))
        return x
