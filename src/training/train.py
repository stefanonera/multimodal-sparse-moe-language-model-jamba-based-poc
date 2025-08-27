import argparse, os, json, time, csv, math, random
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from tqdm import tqdm

from src.datasets.preprocess_image import build_dataloaders
from src.model.jamba_moe import JambaMoEForCIFAR10

# utils


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_yaml(obj, path: Path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


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


# training loop


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

    # logs
    train_loss_csv = logs_dir / "train_loss.csv"
    routing_batch_path = logs_dir / "routing_batch.jsonl"

    autocast = torch.autocast if torch.cuda.is_available() else torch.cpu.amp.autocast
    dtype = torch.bfloat16 if use_bf16 and torch.cuda.is_available() else None

    optimizer.zero_grad(set_to_none=True)
    for bi, batch in enumerate(tqdm(loader, desc=f"Train epoch {epoch_idx}")):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with (
            autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else torch.no_grad() if False else torch.enable_grad()
        ):
            loss, outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = loss / grad_accum_steps

        loss.backward()
        if (bi + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        step_count += 1

        # per-batch logging
        if (bi + 1) % log_every == 0 or (bi + 1) == len(loader):
            avg_loss = total_loss / step_count
            write_csv_row(
                train_loss_csv,
                header=["epoch", "batch", "avg_loss"],
                row=[epoch_idx, bi + 1, avg_loss],
            )

        # routing stats logging (per layer)
        for layer_idx, stats in enumerate(outputs["routing_logs"]):
            entry = {
                "phase": "train",
                "epoch": epoch_idx,
                "batch": bi + 1,
                "layer": layer_idx,
                "counts_per_expert": stats["counts_per_expert"].tolist(),
                "balance_score": float(stats["balance_score"]),
                "topk_entropy": float(stats["topk_entropy"]),
                # kept_mask omitted for size
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
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with (
            autocast(device_type="cuda", dtype=dtype)
            if dtype is not None
            else torch.no_grad()
        ):
            loss, outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        total_loss += loss.item()
        steps += 1

        # routing stats
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="training/config.yaml",
        help="Path to YAML config (default: training/config.yaml)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="base",
        help="Which preset block to use (base, ablation_small, ...)",
    )
    args = parser.parse_args()

    # check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_cfg = cfg.get("base", {})
    data_cfg = cfg.get("data", {})
    enc_cfg = cfg.get("encoders", {})
    fusion_cfg = cfg.get("fusion", {})
    moe_cfg = cfg.get("moe", {})

    if args.preset != "base":
        if args.preset not in cfg:
            raise ValueError(f"Preset '{args.preset}' not found in YAML.")
        preset_override = cfg[args.preset]
        combined = {
            "base": base_cfg,
            "data": data_cfg,
            "encoders": enc_cfg,
            "fusion": fusion_cfg,
            "moe": moe_cfg,
        }
        final = deep_update(
            combined,
            (
                {"base": preset_override}
                if "num_epochs" in preset_override or "output_dir" in preset_override
                else {}
            ),
        )
        for k in ["data", "encoders", "fusion", "moe"]:
            if k in preset_override:
                final[k] = deep_update(final[k], preset_override[k])
        base_cfg, data_cfg, enc_cfg, fusion_cfg, moe_cfg = (
            final["base"],
            final["data"],
            final["encoders"],
            final["fusion"],
            final["moe"],
        )

    seed = int(base_cfg.get("seed", 42))
    set_seed(seed)

    # device & precision
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = (
        bool(base_cfg.get("bf16", False))
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    # run dir
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_cfg.get("output_dir", "checkpoints/poc")) / ts
    ensure_dir(run_dir)
    logs_dir = run_dir / "logs"
    ensure_dir(logs_dir)
    plots_dir = run_dir / "plots"
    ensure_dir(plots_dir)

    # save resolved config
    resolved_cfg = {
        "seed": seed,
        "device": device,
        "bf16": use_bf16,
        "base": base_cfg,
        "data": data_cfg,
        "encoders": enc_cfg,
        "fusion": fusion_cfg,
        "moe": moe_cfg,
    }
    save_yaml(resolved_cfg, run_dir / "resolved_config.yaml")

    # data
    train_loader, val_loader, test_loader = build_dataloaders(
        image_size=int(data_cfg.get("image_size", 224)),
        tokenizer_name=str(
            enc_cfg.get("text", {}).get("tokenizer_name", "ai21labs/Jamba-v0.1")
        ),
        batch_size_train=int(base_cfg.get("per_device_train_batch_size", 64)),
        batch_size_eval=int(base_cfg.get("per_device_eval_batch_size", 64)),
        train_split=str(data_cfg.get("train_split", "train[:90%]")),
        val_split=str(data_cfg.get("val_split", "train[90%:]")),
        test_split=str(data_cfg.get("test_split", "test")),
        num_workers=2,
        text_from_labels=bool(data_cfg.get("text_from_labels", True)),
    )

    # model
    model = JambaMoEForCIFAR10(
        num_classes=10,
        txt_hidden=int(enc_cfg.get("text", {}).get("hidden_size", 2048)),
        img_hidden=int(enc_cfg.get("image", {}).get("proj_out", 2048)),
        model_dim=int(fusion_cfg.get("hidden_size", 2048)),
        ffn_dim=int(moe_cfg.get("ffn_dim", 4096)),
        num_experts=int(moe_cfg.get("num_experts", 4)),
        num_layers=int(moe_cfg.get("num_layers", 1)),
        capacity_factor=float(moe_cfg.get("capacity_factor", 1.25)),
        tokenizer_name=str(
            enc_cfg.get("text", {}).get("tokenizer_name", "ai21labs/Jamba-v0.1")
        ),
    ).to(device)

    # freeze encoders for sanity
    model.freeze_encoders()

    # optimizer (only trainable params)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=float(base_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(base_cfg.get("weight_decay", 0.0)),
    )

    # train loop
    num_epochs = int(base_cfg.get("num_epochs", 2))
    log_every = int(base_cfg.get("log_every", 25))
    grad_accum_steps = int(base_cfg.get("grad_accum_steps", 1))

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
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
        val_loss = evaluate(model, val_loader, device, use_bf16, logs_dir, epoch)

        # save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, run_dir / f"checkpoint_epoch{epoch}.pt")

    # also save a last pointer
    with open(run_dir / "last_run.txt", "w") as f:
        f.write(str(run_dir))

    print(f"Training done. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
