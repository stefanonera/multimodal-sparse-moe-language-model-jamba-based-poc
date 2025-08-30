from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        dataset_name: str = "ag_news",
        split: str = "train[:2000]",
        text_field: str = "text",
        label_field: str = "label",
        tokenizer_name: str = "ai21labs/Jamba-v0.1",
        max_length: int = 128,
    ):
        super().__init__()
        self.ds = load_dataset(dataset_name, split=split)
        self.text_field = text_field
        self.label_field = label_field
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, legacy=False
        )
        self.max_length = max_length
        feat = self.ds.features[label_field]
        self.num_classes = len(getattr(feat, "names", [])) or int(
            self.ds.unique(label_field)
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.ds[idx]
        text = ex[self.text_field]
        label = int(ex[self.label_field])
        toks = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": toks["input_ids"].squeeze(0),
            "attention_mask": toks["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def text_collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_text_dataloaders(
    tokenizer_name: str = "ai21labs/Jamba-v0.1",
    batch_size_train: int = 64,
    batch_size_eval: int = 64,
    dataset_name: str = "ag_news",
    train_split: str = "train[:2000]",
    val_split: str = "test[:1000]",
    max_length: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, int]:
    train_ds = TextClassificationDataset(
        dataset_name=dataset_name,
        split=train_split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )
    val_ds = TextClassificationDataset(
        dataset_name=dataset_name,
        split=val_split,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=text_collate_fn,
    )
    return train_loader, val_loader, train_ds.num_classes
