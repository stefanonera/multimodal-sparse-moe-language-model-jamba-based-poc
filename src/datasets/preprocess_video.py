from typing import Dict, Tuple, List
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_video
from transformers import AutoTokenizer
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int = 224,
        num_frames: int = 8,
        tokenizer_name: str = "ai21labs/Jamba-v0.1",
        max_text_len: int = 16,
        extensions: Tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv"),
    ):
        super().__init__()
        self.root = Path(root)
        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx = {}
        self.idx_to_class = []

        # scan folders
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        for idx, cname in enumerate(classes):
            self.class_to_idx[cname] = idx
            self.idx_to_class.append(cname)
            for p in (self.root / cname).iterdir():
                if p.suffix.lower() in extensions and p.is_file():
                    self.samples.append((p, idx))

        if not self.samples:
            raise FileNotFoundError(f"No video files found under {self.root}")

        self.num_frames = num_frames
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_text_len = max_text_len

        self.tf = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(image_size),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def _caption_from_label(self, y: int) -> str:
        name = self.idx_to_class[y] if 0 <= y < len(self.idx_to_class) else "object"
        return f"a video of a {name}"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, y = self.samples[idx]
        # read_video returns (video[T,H,W,C], audio, info)
        video, _, info = read_video(str(path), pts_unit="sec")
        # video: int tensor [T, H, W, C] in 0..255
        T_total = video.shape[0]
        if T_total == 0:
            raise RuntimeError(f"Empty video: {path}")

        # uniformly sample self.num_frames indices
        if self.num_frames == 1:
            frame_idxs = [T_total // 2]
        else:
            frame_idxs = torch.linspace(0, T_total - 1, self.num_frames).long().tolist()

        frames = video[frame_idxs]  # [F, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [F, C, H, W]
        frames = torch.stack([self.tf(f) for f in frames], dim=0)  # [F, C, H, W]

        caption = self._caption_from_label(y)
        toks = self.tokenizer(
            caption,
            truncation=True,
            padding="max_length",
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        input_ids = toks["input_ids"].squeeze(0)
        attn = toks["attention_mask"].squeeze(0)

        return {
            "video_frames": frames,  # [F, 3, H, W]
            "input_ids": input_ids,  # [T]
            "attention_mask": attn,  # [T]
            "label": torch.tensor(y, dtype=torch.long),
        }


def video_collate_fn(batch):
    # pad/truncate frames to consistent length within batch if needed
    max_f = max(b["video_frames"].shape[0] for b in batch)
    B = len(batch)
    C, H, W = batch[0]["video_frames"].shape[1:]
    frames = torch.zeros(B, max_f, C, H, W, dtype=batch[0]["video_frames"].dtype)
    for i, b in enumerate(batch):
        f = b["video_frames"]
        F = f.shape[0]
        frames[i, :F] = f
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    return {
        "video_frames": frames,  # [B, F, 3, H, W]
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_video_dataloaders(
    root: str,
    image_size: int = 224,
    num_frames: int = 8,
    batch_size_train: int = 4,
    batch_size_eval: int = 4,
    tokenizer_name: str = "ai21labs/Jamba-v0.1",
    max_text_len: int = 16,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = VideoFolderDataset(
        root=os.path.join(root, "train"),
        image_size=image_size,
        num_frames=num_frames,
        tokenizer_name=tokenizer_name,
        max_text_len=max_text_len,
    )
    val_ds = VideoFolderDataset(
        root=os.path.join(root, "val"),
        image_size=image_size,
        num_frames=num_frames,
        tokenizer_name=tokenizer_name,
        max_text_len=max_text_len,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=video_collate_fn,
    )
    return train_loader, val_loader
