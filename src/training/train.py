import argparse, os, json, time, csv, random
from pathlib import Path
from copy import deepcopy

import torch
import torch.optim as optim
import yaml
import numpy as np
from tqdm import tqdm

from src.datasets.preprocess_image import build_dataloaders as build_img_loaders
from src.datasets.preprocess_text import build_text_dataloaders
from src.datasets.preprocess_video import build_video_dataloaders, count_video_classes
from src.model.jamba_moe import JambaMoEClassifier


# utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_csv_row(path: Path, header, row):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


def deep_update(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# per-epoch loops
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    use_bf16,
    log_every,
    logs_dir,
    epoch_idx,
    grad_accum_steps=1,
):
    model.train()
    step_count = 0
    total_loss = 0.0

    train_loss_csv = logs_dir / "train_loss.csv"
    routing_batch_path = logs_dir / "routing_batch.jsonl"
    autocast = torch.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else None

    optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch_idx}")):
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        with (
            autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else torch.enable_grad()
        ):
            loss, outputs = model(
                pixel_values=batch.get("pixel_values"),
                video_frames=batch.get("video_frames"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
            loss = loss / grad_accum_steps
        loss.backward()
        if (bi + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item())
        step_count += 1
        if (bi + 1) % log_every == 0 or (bi + 1) == len(loader):
            avg_loss = total_loss / step_count
            write_csv_row(
                train_loss_csv,
                header=["epoch", "batch", "avg_loss"],
                row=[epoch_idx, bi + 1, avg_loss],
            )

        for layer_idx, stats in enumerate(outputs["routing_logs"]):
            entry = {
                "phase": "train",
                "epoch": epoch_idx,
                "batch": bi + 1,
                "layer": layer_idx,
                "counts_per_expert": stats["counts_per_expert"].tolist(),
                "balance_score": float(stats["balance_score"]),
                "topk_entropy": float(stats["topk_entropy"]),
            }
            with open(routing_batch_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    return total_loss / max(1, step_count)


@torch.no_grad()
def evaluate(model, loader, device, use_bf16, logs_dir, epoch_idx):
    model.eval()
    total_loss = 0.0
    steps = 0
    val_loss_csv = logs_dir / "val_loss.csv"
    routing_batch_path = logs_dir / "routing_batch.jsonl"
    autocast = torch.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else None

    for bi, batch in enumerate(tqdm(loader, desc=f"Valid epoch {epoch_idx}")):
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        with (
            autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else torch.no_grad()
        ):
            loss, outputs = model(
                pixel_values=batch.get("pixel_values"),
                video_frames=batch.get("video_frames"),
                input_ids=batch.get("input_ids"),
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )
        total_loss += float(loss.item())
        steps += 1

        for layer_idx, stats in enumerate(outputs["routing_logs"]):
            entry = {
                "phase": "val",
                "epoch": epoch_idx,
                "batch": bi + 1,
                "layer": layer_idx,
                "counts_per_expert": stats["counts_per_expert"].tolist(),
                "balance_score": float(stats["balance_score"]),
                "topk_entropy": float(stats["topk_entropy"]),
            }
            with open(routing_batch_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    avg_loss = total_loss / max(1, steps)
    write_csv_row(val_loss_csv, header=["epoch", "avg_loss"], row=[epoch_idx, avg_loss])
    return avg_loss


# task builders
def build_task_loaders(cfg):
    task = cfg["base"].get("task", "image_text_cifar10")
    dcfg = cfg["data"]
    txt_tok = cfg["encoders"]["text"].get("tokenizer_name", "ai21labs/Jamba-v0.1")

    if task == "image_text_cifar10":
        train_loader, val_loader, _ = build_img_loaders(
            image_size=int(dcfg["image"].get("image_size", 224)),
            tokenizer_name=txt_tok,
            batch_size_train=int(cfg["base"].get("per_device_train_batch_size", 64)),
            batch_size_eval=int(cfg["base"].get("per_device_eval_batch_size", 64)),
            train_split=str(dcfg["image"].get("train_split", "train[:90%]")),
            val_split=str(dcfg["image"].get("val_split", "train[90%:]")),
            test_split="test",
            num_workers=2,
            text_from_labels=True,
        )
        num_classes = 10
        modalities = ["text", "image"]

    elif task == "text_agnews":
        train_loader, val_loader, num_classes = build_text_dataloaders(
            tokenizer_name=txt_tok,
            batch_size_train=int(cfg["base"].get("per_device_train_batch_size", 64)),
            batch_size_eval=int(cfg["base"].get("per_device_eval_batch_size", 64)),
            dataset_name=str(dcfg["text"].get("dataset_name", "ag_news")),
            train_split=str(dcfg["text"].get("train_split", "train[:2000]")),
            val_split=str(dcfg["text"].get("val_split", "test[:1000]")),
            max_length=int(dcfg["text"].get("max_length", 128)),
            num_workers=2,
        )
        modalities = ["text"]

    elif task == "video_folder":
        train_loader, val_loader = build_video_dataloaders(
            root=str(dcfg["video"].get("root", "data/mini_ucf")),
            image_size=int(dcfg["video"].get("image_size", 224)),
            num_frames=int(dcfg["video"].get("num_frames", 8)),
            batch_size_train=int(dcfg["video"].get("batch_size_train", 4)),
            batch_size_eval=int(dcfg["video"].get("batch_size_eval", 4)),
            tokenizer_name=txt_tok,
            max_text_len=int(dcfg["video"].get("max_text_len", 16)),
            num_workers=2,
        )
        num_classes = count_video_classes(
            str(dcfg["video"].get("root", "data/mini_ucf"))
        )
        modalities = ["text", "video"]
    else:
        raise ValueError(f"Unknown task: {task}")

    return train_loader, val_loader, num_classes, modalities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--preset", type=str, default="base")
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save .pt checkpoint files (disabled by default to save storage)",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_cfg = cfg.get("base", {})
    if args.preset != "base":
        if args.preset not in cfg:
            raise ValueError(f"Preset '{args.preset}' not found.")
        # allow ablation presets to override 'moe' etc.
        for section in ["base", "data", "encoders", "fusion", "moe"]:
            if section in cfg[args.preset]:
                cfg[section] = deep_update(
                    cfg.get(section, {}), cfg[args.preset][section]
                )

    seed = int(cfg["base"].get("seed", 42))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (
        bool(cfg["base"].get("bf16", False))
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    # dirs
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg["base"].get("output_dir", "checkpoints/poc")) / ts
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    ensure_dir(run_dir)
    ensure_dir(logs_dir)
    ensure_dir(plots_dir)

    # data
    train_loader, val_loader, num_classes, modalities = build_task_loaders(cfg)

    # model
    enc = cfg["encoders"]
    fusion = cfg["fusion"]
    moe = cfg["moe"]
    model = JambaMoEClassifier(
        num_classes=num_classes,
        active_modalities=modalities,
        txt_hidden=int(enc["text"].get("hidden_size", 1024)),
        img_hidden=int(enc["image"].get("proj_out", 1024)),
        vid_hidden=int(enc["video"].get("proj_out", 1024)),
        model_dim=int(fusion.get("hidden_size", 1024)),
        ffn_dim=int(moe.get("ffn_dim", 2048)),
        num_experts=int(moe.get("num_experts", 4)),
        num_layers=int(moe.get("num_layers", 1)),
        capacity_factor=float(moe.get("capacity_factor", 1.25)),
        tokenizer_name=str(enc["text"].get("tokenizer_name", "ai21labs/Jamba-v0.1")),
        freeze_encoders=True,
    ).to(device)
    model.freeze_encoders()

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=float(cfg["base"].get("learning_rate", 5e-4)),
        weight_decay=float(cfg["base"].get("weight_decay", 0.0)),
    )

    # train
    num_epochs = int(cfg["base"].get("num_epochs", 2))
    log_every = int(cfg["base"].get("log_every", 25))
    grad_accum_steps = int(cfg["base"].get("grad_accum_steps", 1))
    save_checkpoints = args.save_checkpoints or bool(
        cfg["base"].get("save_checkpoints", False)
    )

    for epoch in range(1, num_epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_bf16,
            log_every,
            logs_dir,
            epoch,
            grad_accum_steps,
        )
        va = evaluate(model, val_loader, device, use_bf16, logs_dir, epoch)

        # Only save checkpoints if explicitly requested
        if save_checkpoints:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "train_loss": tr,
                    "val_loss": va,
                },
                run_dir / f"checkpoint_epoch{epoch}.pt",
            )

    (run_dir / "last_run.txt").write_text(str(run_dir))
    with open(run_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    if save_checkpoints:
        print(f"Done. Run dir: {run_dir} (with checkpoints)")
    else:
        print(f"Done. Run dir: {run_dir} (no checkpoints saved)")


if __name__ == "__main__":
    main()
