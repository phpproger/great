import os
import glob
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer(custom_path: Optional[str] = None):
    """Loads a fast LLaMA / BPE tokenizer.
    Prefers local cached SmolLM2 tokenizer if available, falls back to AutoTokenizer or GPT2.
    """
    if custom_path and os.path.exists(custom_path):
        return AutoTokenizer.from_pretrained(custom_path, local_files_only=True)

    # Check local HuggingFace cache for SmolLM2 (LLaMA architecture tokenizer)
    smollm2_pattern = os.path.expanduser("~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*")
    snapshots = glob.glob(smollm2_pattern)
    if snapshots and os.path.exists(snapshots[0]):
        tokenizer = AutoTokenizer.from_pretrained(snapshots[0], local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    # Fallback to online or local GPT2 / SmolLM2
    for model_name in ["HuggingFaceTB/SmolLM2-135M", "gpt2"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception:
            continue

    raise RuntimeError("Could not find or load a compatible tokenizer.")


class TinyStoriesDataset(Dataset):
    """PyTorch Dataset for TinyStories.
    Supports tokenizing on-the-fly or pre-tokenizing a subset of stories.
    """
    def __init__(
        self,
        split: str = "train",
        max_length: int = 256,
        num_samples: Optional[int] = 500,
        tokenizer = None,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

        # Load cached roneneldan/TinyStories dataset safely without network requests
        try:
            full_ds = load_dataset("roneneldan/TinyStories", split=split)
            if num_samples is not None and num_samples < len(full_ds):
                self.dataset = full_ds.select(range(num_samples))
            else:
                self.dataset = full_ds
        except Exception:
            split_str = f"{split}[:{num_samples}]" if num_samples is not None else split
            self.dataset = load_dataset("roneneldan/TinyStories", split=split_str)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        text = item["text"]

        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        # Create labels: mask padding tokens with -100 so loss ignores them
        labels = input_ids.clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "text": text,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "texts": texts,
    }


class WikiTextDataset(Dataset):
    """Memory-mapped pretokenized WikiText-103 dataset.
    Provides fast, zero-copy access to 116M+ tokens with article boundaries.
    """
    def __init__(
        self,
        data_dir: str = "data/wikitext",
        split: str = "train",
        seq_len: int = 32,
        pairs: bool = False,
        max_samples: Optional[int] = None,
        stride: Optional[int] = None,
    ):
        self.seq_len = seq_len
        self.pairs = pairs
        self.window = 2 * seq_len if pairs else seq_len
        self.stride = stride if stride is not None else self.window

        offsets_file = os.path.join(data_dir, f"{split}-offsets.npy")
        tokens_file = os.path.join(data_dir, f"{split}-tokens.bin")

        if not os.path.exists(tokens_file):
            raise FileNotFoundError(f"Missing wikitext tokens file at {tokens_file}")

        self.offsets = np.load(offsets_file)
        self.tokens = np.memmap(tokens_file, dtype="<u4", mode="r")

        # Build list of valid (start, end) coordinate windows within articles
        self.samples = []
        num_articles = len(self.offsets)
        for i in range(num_articles):
            art_start = int(self.offsets[i])
            art_end = int(self.offsets[i + 1]) if i + 1 < num_articles else len(self.tokens)
            art_len = art_end - art_start
            if art_len < self.window:
                continue
            for pos in range(art_start, art_end - self.window + 1, self.stride):
                self.samples.append(pos)
                if max_samples and len(self.samples) >= max_samples:
                    break
            if max_samples and len(self.samples) >= max_samples:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.samples[idx]
        block = torch.from_numpy(self.tokens[start : start + self.window].astype(np.int64))

        if self.pairs:
            x = block[:self.seq_len]
            y = block[self.seq_len : 2 * self.seq_len]
            return {"context": x, "continuation": y}
        else:
            return {"input_ids": block}


def get_wikitext_dataloader(
    data_dir: str = "data/wikitext",
    split: str = "train",
    batch_size: int = 16,
    seq_len: int = 32,
    pairs: bool = False,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = WikiTextDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        pairs=pairs,
        max_samples=max_samples,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



def get_tinystories_dataloader(
    split: str = "train",
    batch_size: int = 4,
    max_length: int = 256,
    num_samples: Optional[int] = 500,
    tokenizer = None,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience helper to create a DataLoader for TinyStories."""
    dataset = TinyStoriesDataset(
        split=split,
        max_length=max_length,
        num_samples=num_samples,
        tokenizer=tokenizer,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
