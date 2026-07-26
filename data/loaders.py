"""
Dataloader factory — download, tokenize, pack, and serve WikiText-103.
"""

import os
import torch
from torch.utils.data import DataLoader

from config import TransformerConfig
from .dataset import PackedTextDataset


def _tokenize_and_pack(
    split: str, tokenizer, seq_len: int, cache_dir: str = "data_cache"
) -> PackedTextDataset:
    from datasets import load_dataset

    cache_path = os.path.join(cache_dir, f"wikitext103_{split}_tokens.pt")

    if os.path.exists(cache_path):
        print(f"  Loading cached tokens from {cache_path}")
        token_ids = torch.load(cache_path, weights_only=True).tolist()
    else:
        print(f"  Downloading & tokenizing WikiText-103 ({split})...")
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
        except Exception:
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

        eos_id = tokenizer.eos_token_id
        all_ids = []
        for example in ds:
            text = example["text"].strip()
            if text:
                ids = tokenizer.encode(text)
                all_ids.extend(ids)
                all_ids.append(eos_id)

        os.makedirs(cache_dir, exist_ok=True)
        torch.save(torch.tensor(all_ids, dtype=torch.long), cache_path)
        token_ids = all_ids
        print(f"  Cached {len(token_ids):,} tokens to {cache_path}")

    return PackedTextDataset(token_ids, seq_len)


def get_dataloaders(config: TransformerConfig) -> dict:
    from transformers import AutoTokenizer

    print("Setting up data pipeline...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.encode("<|endoftext|>")[0]

    # Training data
    print(f"  Preparing training data (seq_len={config.train_seq_len})...")
    train_ds = _tokenize_and_pack("train", tokenizer, config.train_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size // config.grad_accum_steps,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    loaders = {"train": train_loader}

    # Evaluation data at each eval seq_len
    for eval_len in config.eval_seq_lens:
        print(f"  Preparing eval data (seq_len={eval_len})...")
        val_ds = _tokenize_and_pack("validation", tokenizer, eval_len)
        eval_batch_size = max(1, 8 if eval_len >= 1024 else (16 if eval_len >= 768 else 32))
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        loaders[f"val_{eval_len}"] = val_loader

    print(f"  Train: {len(train_ds):,} blocks of {config.train_seq_len} tokens")
    for eval_len in config.eval_seq_lens:
        key = f"val_{eval_len}"
        print(f"  Eval ({eval_len}): {len(loaders[key].dataset):,} blocks")

    return loaders
