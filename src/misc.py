import torch
import os
import argparse
import json
from . import network
from . import losses
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


def build_training_stack(opts, compile_model=True, device="cuda:0"):
    """
    Builds the training stack: model, optimizer, scheduler, step counter,
    best metric, validation history. Resumes from latest.pth if opts.resume=True.
    """
    # --- Initialize model ---
    model = network.Unet(
        dimension=opts.dimension,
        input_nc=opts.in_channels,
        output_nc=opts.out_channels,
        num_downs=opts.num_levels,
        ngf=opts.base_filters,
        norm="batch",
    ).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay, fused=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=opts.max_steps
    )

    # step counter, best metric, val history
    step = 0
    best_metric = 0.0
    val_history = {"step": [], "mean_dice": [], "per_class_dice": []}

    # resume, if needed
    checkpoint_path = os.path.join(opts.output_dir, "latest.pth")
    if opts.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        state = checkpoint["model_state_dict"]
        clean_state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        step = checkpoint.get("step", 0)
        best_metric = checkpoint.get("best_metric", 0.0)
        val_history = checkpoint.get("val_history", val_history)

    # compile
    if compile_model:
        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=True)

    return model, optimizer, scheduler, step, best_metric, val_history


def save_checkpoint(
    opts: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    step: int,
    best_metric: float,
    save_dir: str,
    val_history: dict,
    filename: str = "checkpoint.pth",
):
    """Saves the training checkpoint"""
    model_unwrapped = model._orig_mod if hasattr(model, "_orig_mod") else model

    checkpoint = {
        "model_state_dict": model_unwrapped.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "args": vars(opts),
        "val_history": val_history,
        "wandb_run_id": wandb.run.id if wandb_available and wandb.run else None,
    }
    os.makedirs(save_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(save_dir, filename))

    # save the validation history
    if filename == "latest.pth":
        json_path = os.path.join(save_dir, "val_history.json")
        with open(json_path, "w") as f:
            json.dump(val_history, f)

def minmax_norm(x, eps=1e-6):
    x_min = x.amin(dim=(1,2,3,4), keepdim=True)
    x_max = x.amax(dim=(1,2,3,4), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def is_wandb_initialized():
    return wandb_available and wandb.run is not None


def setup_training(opts: argparse.Namespace):
    pass


def wandb_log(opts: argparse.Namespace, dice_scores: dict, step: int, log_val: float):
    row = {
        "step": step,
        "mean_dice": dice_scores["mean_dice"],
    }
    for class_name, stats in dice_scores["per_class_dice"].items():
        row[f"dice_{class_name}"] = stats["dice"]
        row[f"count_{class_name}"] = stats["count"]

    # log scalars as before
    wandb.log(
        {f"validation/{opts.dataset}/mean_dice": dice_scores["mean_dice"]}, step=step
    )
    for class_name, stats in dice_scores["per_class_dice"].items():
        wandb.log(
            {
                f"validation/{opts.dataset}/{class_name}": stats["dice"],
                # f"validation/{opts.dataset}/{class_name}_count": stats["count"],
            },
            step=step,
        )

    # log the best validation loss so far
    wandb.log({f"validation/{opts.dataset}/best_val": log_val}, step=step)


def load_feature_extractor(opts: argparse.Namespace):
    if opts.method == "dropgen":
        feat_ex = network.Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=4,
            ngf=opts.feature_dim,
            layer_index=opts.layer_index,
            norm=opts.norm_type,
        )
        feat_ex.load_state_dict(
            torch.load(
                ".../pretrained/anatomix/anatomix.pth",
                map_location="cpu",
            ),
            strict=True,
        )
        feat_ex = torch.nn.Sequential(feat_ex, torch.nn.InstanceNorm3d(16))
        feat_ex.eval()
        return feat_ex
    else:
        return None

def get_loss_fn(opts: argparse.Namespace):
    return losses.Loss(opts)

