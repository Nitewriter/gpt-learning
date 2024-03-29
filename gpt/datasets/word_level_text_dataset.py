from collections import Counter
from pathlib import Path

import torch
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset


class WordLevelTextDataset(Dataset):
    def __init__(self, file_path: Path, seq_length: int):
        self.seq_length = seq_length
        self.vocab, self.vocab_inv = self.build_vocab(file_path)
        self.token_ids = self.tokenize_file(file_path)
        self.num_tokens = len(self.token_ids)

    def build_vocab(self, file_path: Path):
        vocab_counts = Counter()
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                tokens = word_tokenize(line.lower())
                vocab_counts.update(tokens)

        vocab = {
            token: idx for idx, (token, _) in enumerate(vocab_counts.most_common())
        }
        vocab_inv = {idx: token for token, idx in vocab.items()}
        return vocab, vocab_inv

    def tokenize_file(self, file_path: Path):
        token_ids = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                tokens = word_tokenize(line.lower())
                line_token_ids = [
                    self.vocab[token] for token in tokens if token in self.vocab
                ]
                token_ids.extend(line_token_ids)
        return token_ids

    def __len__(self):
        return self.num_tokens - self.seq_length

    def __getitem__(self, idx: int):
        start_idx = idx
        end_idx = idx + self.seq_length + 1
        input_seq = self.token_ids[start_idx:end_idx]
        x = torch.tensor(input_seq[:-1], dtype=torch.long)
        y = torch.tensor(input_seq[1:], dtype=torch.long)
        return x, y
