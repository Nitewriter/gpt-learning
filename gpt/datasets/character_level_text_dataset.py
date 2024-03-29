import torch
from torch.utils.data import Dataset
from pathlib import Path


class CharacterLevelTextDataset(Dataset):
    def __init__(self, file_path: Path, seq_length: int):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.token_ids = self.tokenize_file(file_path)
        self.num_tokens = len(self.token_ids)
        self.vocab_size = len(self.char_to_idx)

    def tokenize_file(self, file_path: Path):
        token_ids = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.lower()  # Convert to lowercase for consistency
                for char in line:
                    if char not in self.char_to_idx:
                        idx = len(self.char_to_idx)
                        self.char_to_idx[char] = idx
                        self.idx_to_char[idx] = char
                    token_ids.append(self.char_to_idx[char])
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
