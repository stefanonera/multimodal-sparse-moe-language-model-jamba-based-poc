from pathlib import Path
import json
import csv
from typing import List, Dict, Tuple, Any, DefaultDict
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_csv(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: float(v) if k != "phase" else v for k, v in r.items()})
    return rows


def aggregate_counts_per_epoch(
    batch_entries: List[Dict], num_experts: int
) -> Dict[Tuple[int, int, str], np.ndarray]:
    """
    Returns {(epoch, layer, phase): counts[E]} aggregated over batches
    """
    buckets: DefaultDict[Tuple[int, int, str], np.ndarray] = defaultdict(
        lambda: np.zeros((num_experts,), dtype=np.int64)
    )
    for e in batch_entries:
        epoch = int(e["epoch"])
        layer = int(e["layer"])
        phase = e["phase"]
        counts = np.array(e["counts_per_expert"], dtype=np.int64)
        buckets[(epoch, layer, phase)] += counts
    return buckets


def series_from_batch(
    entries: List[Dict], layer: int, phase: str
) -> Dict[str, List[float]]:
    bs, ent = [], []
    x = []
    for e in entries:
        if e["layer"] == layer and e["phase"] == phase:
            x.append(e["batch"])
            bs.append(e["balance_score"])
            ent.append(e["topk_entropy"])
    order = np.argsort(x)
    return {
        "x": list(np.array(x)[order]),
        "balance": list(np.array(bs)[order]),
        "entropy": list(np.array(ent)[order]),
    }


def plot_loss_curves(train_csv: Path, val_csv: Path, out_path: Path):
    train_rows = []
    if train_csv.exists():
        with open(train_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                train_rows.append(
                    {
                        "epoch": int(row["epoch"]),
                        "batch": int(row["batch"]),
                        "avg_loss": float(row["avg_loss"]),
                    }
                )
    val_rows = []
    if val_csv.exists():
        with open(val_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                val_rows.append(
                    {"epoch": int(row["epoch"]), "avg_loss": float(row["avg_loss"])}
                )

    plt.figure()
    if train_rows:
        xs = [f"{d['epoch']}.{d['batch']}" for d in train_rows]
        ys = [d["avg_loss"] for d in train_rows]
        plt.plot(range(len(xs)), ys, label="train avg_loss")
    if val_rows:
        xs_v = [d["epoch"] for d in val_rows]
        ys_v = [d["avg_loss"] for d in val_rows]
        plt.plot(xs_v, ys_v, marker="o", label="val avg_loss")
    plt.xlabel("step / epoch")
    plt.ylabel("loss")
    plt.title("Loss curves")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_balance_entropy(
    batch_entries: List[Dict], num_layers: int, out_dir: Path, phase: str = "train"
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer in range(num_layers):
        s = series_from_batch(batch_entries, layer=layer, phase=phase)
        if not s["x"]:
            continue
        # balance
        plt.figure()
        plt.plot(s["x"], s["balance"])
        plt.xlabel("batch")
        plt.ylabel("load balance score")
        plt.title(f"Load balance (layer {layer}, {phase})")
        plt.savefig(out_dir / f"balance_layer{layer}_{phase}.png", bbox_inches="tight")
        plt.close()

        # entropy
        plt.figure()
        plt.plot(s["x"], s["entropy"])
        plt.xlabel("batch")
        plt.ylabel("top-k entropy")
        plt.title(f"Top-k entropy (layer {layer}, {phase})")
        plt.savefig(out_dir / f"entropy_layer{layer}_{phase}.png", bbox_inches="tight")
        plt.close()


def plot_expert_usage_bars(counts: np.ndarray, out_path: Path, title: str):
    plt.figure()
    x = np.arange(len(counts))
    plt.bar(x, counts)
    plt.xlabel("expert id")
    plt.ylabel("count")
    plt.title(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_counts_heatmap(
    batch_entries: List[Dict],
    num_experts: int,
    epoch: int,
    layer: int,
    phase: str,
    out_path: Path,
):
    # build [num_batches, num_experts] matrix for that epoch/layer/phase
    batches = sorted(
        set(
            int(e["batch"])
            for e in batch_entries
            if e["epoch"] == epoch and e["layer"] == layer and e["phase"] == phase
        )
    )
    if not batches:
        return
    mat = np.zeros((len(batches), num_experts), dtype=np.int64)
    idx_map = {b: i for i, b in enumerate(batches)}
    for e in batch_entries:
        if e["epoch"] == epoch and e["layer"] == layer and e["phase"] == phase:
            i = idx_map[int(e["batch"])]
            counts = np.array(e["counts_per_expert"], dtype=np.int64)
            mat[i] = counts

    plt.figure()
    plt.imshow(mat, aspect="auto", interpolation="nearest")
    plt.colorbar(label="tokens kept")
    plt.xlabel("expert id")
    plt.ylabel("batch")
    plt.title(f"Expert usage heatmap (epoch {epoch}, layer {layer}, {phase})")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_expert_usage_epochs_overview(
    routing_batch, num_experts, num_layers, out_path, phase="train"
):
    """
    Plot expert usage evolution across epochs for all layers in a single overview plot.
    Creates subplots for each layer showing expert usage trends over epochs.
    """
    # Aggregate expert counts by epoch and layer
    epoch_layer_counts = {}
    for entry in routing_batch:
        if entry["phase"] != phase:
            continue
        epoch = entry["epoch"]
        layer = entry["layer"]
        counts = np.array(entry["counts_per_expert"])

        key = (epoch, layer)
        if key not in epoch_layer_counts:
            epoch_layer_counts[key] = []
        epoch_layer_counts[key].append(counts)

    # Average counts per epoch-layer
    epoch_layer_avg = {}
    for (epoch, layer), count_list in epoch_layer_counts.items():
        epoch_layer_avg[(epoch, layer)] = np.mean(count_list, axis=0)

    # Create subplots for each layer
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]

    epochs = sorted(set(epoch for epoch, _ in epoch_layer_avg.keys()))

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        # Prepare data for this layer
        layer_data = np.zeros((len(epochs), num_experts))
        for i, epoch in enumerate(epochs):
            if (epoch, layer_idx) in epoch_layer_avg:
                layer_data[i] = epoch_layer_avg[(epoch, layer_idx)]

        # Plot expert usage trends
        for expert_idx in range(num_experts):
            ax.plot(
                epochs,
                layer_data[:, expert_idx],
                marker="o",
                label=f"Expert {expert_idx}",
                linewidth=2,
            )

        ax.set_title(f"Expert Usage Evolution - Layer {layer_idx} ({phase.upper()})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Average Expert Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_expert_heatmap_epochs_overview(
    routing_batch, num_experts, num_layers, out_path, phase="train"
):
    """
    Plot expert usage heatmaps across epochs for all layers in a single overview plot.
    Creates a grid showing how expert usage patterns change over epochs.
    """
    # Aggregate expert counts by epoch and layer
    epoch_layer_counts = {}
    for entry in routing_batch:
        if entry["phase"] != phase:
            continue
        epoch = entry["epoch"]
        layer = entry["layer"]
        counts = np.array(entry["counts_per_expert"])

        key = (epoch, layer)
        if key not in epoch_layer_counts:
            epoch_layer_counts[key] = []
        epoch_layer_counts[key].append(counts)

    # Average counts per epoch-layer
    epoch_layer_avg = {}
    for (epoch, layer), count_list in epoch_layer_counts.items():
        epoch_layer_avg[(epoch, layer)] = np.mean(count_list, axis=0)

    epochs = sorted(set(epoch for epoch, _ in epoch_layer_avg.keys()))

    # Create a large heatmap: rows=layers, cols=epochs, each cell shows expert distribution
    fig, axes = plt.subplots(
        num_layers, len(epochs), figsize=(3 * len(epochs), 3 * num_layers)
    )
    if num_layers == 1 and len(epochs) == 1:
        axes = [[axes]]
    elif num_layers == 1:
        axes = [axes]
    elif len(epochs) == 1:
        axes = [[ax] for ax in axes]

    for layer_idx in range(num_layers):
        for epoch_idx, epoch in enumerate(epochs):
            ax = axes[layer_idx][epoch_idx]

            if (epoch, layer_idx) in epoch_layer_avg:
                data = epoch_layer_avg[(epoch, layer_idx)].reshape(1, -1)
            else:
                data = np.zeros((1, num_experts))

            im = ax.imshow(data, cmap="viridis", aspect="auto")
            ax.set_title(f"L{layer_idx} E{epoch}")
            ax.set_xticks(range(num_experts))
            ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
            ax.set_yticks([])

            # Add colorbar for each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add overall title
    fig.suptitle(f"Expert Usage Heatmap Evolution - {phase.upper()}", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
