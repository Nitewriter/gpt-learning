import torch
import torch.nn as nn

from .transformer_block import TransformerBlock


class GPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        num_heads,
        num_layers,
        hidden_size,
        dropout=0.1,
    ):
        super(GPT2, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.positional_embedding = nn.Embedding(
            1000, embedding_size
        )  # Assuming maximum sequence length of 1000
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, hidden_size, dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size)

    def forward(self, input_ids):
        token_embeddings = self.token_embedding(input_ids)
        batch_size, seq_length = input_ids.size()
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        positional_embeddings = self.positional_embedding(position_ids)

        embeddings = token_embeddings + positional_embeddings
        for transformer_block in self.transformer_blocks:
            embeddings = transformer_block(embeddings)

        embeddings = self.ln(embeddings)
        logits = self.fc(embeddings)
        return logits
