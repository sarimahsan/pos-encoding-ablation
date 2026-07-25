"""
PackedTextDataset — concatenate documents with EOS, chunk into fixed blocks.

Standard causal-LM packing: no padding tokens, no wasted compute.
Each item is (input_ids, targets) where targets = input_ids shifted by 1.
"""

import torch
from torch.utils.data import Dataset


class PackedTextDataset(Dataset):
    """
    Pack tokenized text into fixed-length blocks for causal LM training.

    Documents are concatenated with EOS separators, then chunked into
    non-overlapping blocks of (seq_len + 1). The +1 provides the
    target for the last position.
    """

    def __init__(self, token_ids: list, seq_len: int):
        self.seq_len = seq_len
        block_size = seq_len + 1
        n_blocks = len(token_ids) // block_size
        self.data = torch.tensor(
            token_ids[:n_blocks * block_size], dtype=torch.long
        ).view(n_blocks, block_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        block = self.data[idx]
        input_ids = block[:-1]   # first seq_len tokens
        targets = block[1:]      # shifted by 1
        return input_ids, targets
