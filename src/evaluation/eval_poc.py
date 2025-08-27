import argparse
from pathlib import Path
import yaml
import numpy as np

from .metrics import (
    load_jsonl,
    plot_loss_curves,
    aggregate_counts_per_epoch,
    plot_expert_usage_bars,
    plot_counts_heatmap,
    plot_balance_entropy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir", type=str, required=True, help="Path to a single training run dir"
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # load resolved config to know num_layers/experts
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found in {run_dir}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    num_layers = int(cfg["moe"].get("num_layers", 1))
    num_experts = int(cfg["moe"].get("num_experts", 4))

    # 1. loss curves
    plot_loss_curves(
        train_csv=logs_dir / "train_loss.csv",
        val_csv=logs_dir / "val_loss.csv",
        out_path=plots_dir / "loss_curves.png",
    )

    # 2. routing batch stats
    routing_batch = load_jsonl(logs_dir / "routing_batch.jsonl")

    # 2a. load balance & entropy over batches, per layer
    plot_balance_entropy(
        routing_batch,
        num_layers=num_layers,
        out_dir=plots_dir / "balance_entropy_train",
        phase="train",
    )
    plot_balance_entropy(
        routing_batch,
        num_layers=num_layers,
        out_dir=plots_dir / "balance_entropy_val",
        phase="val",
    )

    # 2b. aggregate counts per epoch (train and val separate)
    buckets = aggregate_counts_per_epoch(routing_batch, num_experts=num_experts)
    for (epoch, layer, phase), counts in buckets.items():
        plot_expert_usage_bars(
            counts=counts,
            out_path=plots_dir / f"expert_usage_epoch{epoch}_layer{layer}_{phase}.png",
            title=f"Expert counts (epoch {epoch}, layer {layer}, {phase})",
        )
        # heatmap across batches for that epoch-layer-phase
        plot_counts_heatmap(
            batch_entries=routing_batch,
            num_experts=num_experts,
            epoch=epoch,
            layer=layer,
            phase=phase,
            out_path=plots_dir / f"heatmap_epoch{epoch}_layer{layer}_{phase}.png",
        )

    print(f"Eval complete. Plots in {plots_dir}")


if __name__ == "__main__":
    main()
