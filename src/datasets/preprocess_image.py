from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer
import random

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CIFAR10WithCaptions(Dataset):
    def __init__(
        self,
        split: str = "train",
        image_size: int = 224,
        tokenizer_name: str = "ai21labs/Jamba-v0.1",
        text_from_labels: bool = True,
        max_text_len: int = 32,
    ):
        super().__init__()
        self.ds = load_dataset("cifar10", split=split)
        self.label_names = self.ds.features["label"].names
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True, legacy=False
        )
        self.text_from_labels = text_from_labels
        self.max_text_len = max_text_len

        self.tf = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ToTensor(),  # convert PIL to tensor and normalize to [0,1]
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        # add variety to captions to prevent overfitting
        self.caption_templates = [
            "a photo of a {name}",
            "an image showing a {name}",
            "this is a {name}",
            "a picture of a {name}",
            "a {name} in the image",
        ]

    def __len__(self):
        return len(self.ds)

    def _caption_from_label(self, y: int) -> str:
        name = self.label_names[y]
        # add some variety to prevent the model from memorizing exact patterns
        template = random.choice(self.caption_templates)
        return template.format(name=name)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.ds[idx]
        img = item["img"]  # PIL Image
        y = int(item["label"])

        pixel_values = self.tf(img)
        caption = self._caption_from_label(y) if self.text_from_labels else "a photo"

        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        # squeeze batch dim
        input_ids = toks["input_ids"].squeeze(0)
        attn_mask = toks["attention_mask"].squeeze(0)

        return {
            "pixel_values": pixel_values,  # [3, H, W]
            "input_ids": input_ids,  # [T]
            "attention_mask": attn_mask,  # [T]
            "label": torch.tensor(y, dtype=torch.long),
        }


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataloaders(
    image_size: int,
    tokenizer_name: str,
    batch_size_train: int,
    batch_size_eval: int,
    train_split: str = "train[:90%]",
    val_split: str = "train[90%:]",
    test_split: str = "test",
    num_workers: int = 2,
    text_from_labels: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CIFAR10WithCaptions(
        split=train_split,
        image_size=image_size,
        tokenizer_name=tokenizer_name,
        text_from_labels=text_from_labels,
    )
    val_ds = CIFAR10WithCaptions(
        split=val_split,
        image_size=image_size,
        tokenizer_name=tokenizer_name,
        text_from_labels=text_from_labels,
    )
    test_ds = CIFAR10WithCaptions(
        split=test_split,
        image_size=image_size,
        tokenizer_name=tokenizer_name,
        text_from_labels=text_from_labels,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
