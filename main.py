import os
import torch
import tqdm
import options
import monai
import argparse
import warnings
import numpy as np
from src import data
from src import misc
from src import network

# suppress specific PyTorch deprecation warnings
warnings.filterwarnings(
    "ignore", message=".*non-tuple sequence for multidimensional indexing.*"
)

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False


def main(opts: argparse.Namespace):
    # buld training stack
    unet, optimizer, lr_scheduler, step_cnt, best_metric, val_history = (
        misc.build_training_stack(opts, compile_model=True, device=opts.device)
    )

    # feature extractor
    feat_ex = misc.load_feature_extractor(opts)
    if feat_ex is not None:
        feat_ex = feat_ex.to(opts.device)
        feat_ex = torch.compile(
            feat_ex, mode="max-autotune", fullgraph=False, dynamic=True
        )
        feat_ex.eval()  # set to eval mode

    # loss
    criterion = misc.get_loss_fn(opts)

    # dataloader
    dataloaders = data.get_dataloaders(opts)

    # initialize pbar
    pbar = tqdm.tqdm(total=opts.max_steps, initial=step_cnt)
    while step_cnt < opts.max_steps:
        for batch in dataloaders["train"]:
            optimizer.zero_grad()
            if step_cnt + 1 > opts.max_steps:
                break

            # move to device
            inputs = batch["image"].to(opts.device, non_blocking=True)
            labels = batch["label"].to(opts.device, non_blocking=True)
            
            # mixed precision forward/backward
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):

                # if using `Stable Representations Enable Domain Generalized Biomedical Image Segmentation`
                if feat_ex is not None:
                    with torch.no_grad():
                        feat_maps = feat_ex(inputs)
                        inputs = torch.cat([inputs, feat_maps], dim=1) # concatenate feature maps to inputs
                        inputs = torch.nn.functional.dropout3d(
                            inputs, p=opts.dropout_prob, training=True
                        )  # some dropout for regularization

                outputs = unet(inputs)
                losses = criterion(outputs, labels)
                loss = losses['loss']

            # backprop + optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            step_cnt += 1

            # updating pbar + logging
            if step_cnt % 10 == 0:
                pbar.update(10)
                pbar.set_description(
                    f"Step {step_cnt}/{opts.max_steps} Loss: {loss.item():.4f}"
                )
                if wandb_available and misc.is_wandb_initialized():
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": lr_scheduler.get_last_lr()[0],
                            **{f"train/{k}": v.item() for k, v in losses.items() if k != 'loss'}
                        },
                        step=step_cnt,
                    )

            # validation
            if step_cnt % opts.eval_interval == 0:
                scores = validate(
                    opts,
                    unet,
                    feat_ex,
                    dataloaders["val"],
                    step=step_cnt,
                    best_metric=best_metric,
                )
                if scores["mean_dice"] > best_metric:
                    print(
                        f'Saving new best model at step {step_cnt} with mean dice {scores["mean_dice"]:.4f} over previous best {best_metric:.4f}.'
                    )
                    best_metric = scores["mean_dice"]
                    misc.save_checkpoint(
                        opts,
                        unet,
                        optimizer,
                        lr_scheduler,
                        step_cnt,
                        best_metric,
                        opts.output_dir,
                        val_history,
                        filename="best.pth",
                    )
                val_history["step"].append(step_cnt)
                val_history["mean_dice"].append(scores["mean_dice"])
                val_history["per_class_dice"].append(scores["per_class_dice"])

                # save latest checkpoint
                misc.save_checkpoint(
                    opts,
                    unet,
                    optimizer,
                    lr_scheduler,
                    step_cnt,
                    best_metric,
                    opts.output_dir,
                    val_history,
                    filename="latest.pth",
                )

    pbar.close()

    test(opts, unet, feat_ex, dataloaders["test"])


@torch._dynamo.disable
def sliding_window_wrapper(inputs, model, crop_size, sw_batch_size):
    """Wrapper for MONAI sliding window inference; compatibility with torch.compile"""
    return monai.inferers.sliding_window_inference(
        inputs=inputs,
        roi_size=crop_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=0.5,
    )

@torch.no_grad()
def validate(
    opts: argparse.Namespace,
    model: torch.nn.Module,
    feat_ex: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    step: int = 0,
    best_metric: float = 0.0,
    wandb_log: bool = True,
):
    """
    Validation with sliding window inference for large 3D volumes.
    More accurate but slower than standard validation.
    """

    class_names = {v: k for k, v in opts.class_mapping.items()}
    foreground_classes = [class_names[i] for i in range(1, opts.num_classes)]

    model.eval()

    dice_metric = monai.metrics.DiceMetric(
        include_background=False, reduction="mean_batch", get_not_nans=True
    )

    if opts.method == "dropgen" and feat_ex is not None:
        model = network.InferenceExtractor(opts, feat_ex, model)

    dice_metric.reset()
    pbar = tqdm.tqdm(
        total=len(dataloader), desc=f"Validation at step {step}", leave=False
    )

    for batch in dataloader:
        with torch.no_grad():
            inputs = batch["image"].to(opts.device, non_blocking=True)
            labels = batch["label"].to(opts.device, non_blocking=True).squeeze(1)
            
            if "gin" in opts.method:
                # stack the image channel-wise to 3
                inputs = inputs.repeat_interleave(3, dim=1)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = sliding_window_wrapper(
                    inputs,
                    model,
                    crop_size=opts.crop_size,
                    sw_batch_size=opts.batch_size,
                )

            outputs = torch.argmax(outputs, dim=1)

            one_hot_preds = (
                torch.nn.functional.one_hot(
                    outputs.long(), num_classes=opts.num_classes
                )
                .permute(0, 4, 1, 2, 3)
                .float()
            )
            one_hot_labels = (
                torch.nn.functional.one_hot(labels.long(), num_classes=opts.num_classes)
                .permute(0, 4, 1, 2, 3)
                .float()
            )

            dice_metric(y_pred=one_hot_preds, y=one_hot_labels)

            pbar.update(1)

    pbar.close()
    result = dice_metric.aggregate()

    if isinstance(result, tuple):
        mean_dice, not_nans = result  # shape: (C-1,)
    else:
        mean_dice = result
        not_nans = torch.ones_like(mean_dice)

    # Build results dictionary
    dice_scores = {"mean_dice": mean_dice.mean().item(), "per_class_dice": {}}

    print(
        f'Validation Mean Dice: {dice_scores["mean_dice"]:.4f} (Best: {best_metric:.4f})'
    )

    for idx, class_name in enumerate(foreground_classes):
        dice_scores["per_class_dice"][class_name] = {
            "dice": mean_dice[idx].item(),
            "count": int(not_nans[idx].item()),
        }

    dice_metric.reset()
    model.train()
    log_val = np.copy(best_metric)
    # log with wandb
    if wandb_available and wandb_log and misc.is_wandb_initialized():
        if dice_scores["mean_dice"] > best_metric:
            wandb.run.summary["best_validation_mean_dice"] = dice_scores["mean_dice"]
            log_val = dice_scores["mean_dice"]

        misc.wandb_log(opts, dice_scores, step, log_val)

    return dice_scores


@torch.no_grad()
def test(
    opts: argparse.Namespace,
    model: torch.nn.Module,
    feat_ex: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
):
    print("Running on test set...")

    # we still get global metrics for test set with validate()
    scores = validate(
        opts,
        model,
        feat_ex,
        dataloader,
        step=0,
        best_metric=0.0,
        wandb_log=False,
    )

    # per-subject analysis
    per_subject = []
    dice_metric = monai.metrics.DiceMetric(
        include_background=False,
        reduction="none",          # import for per-subject analysis
        get_not_nans=True
    )

    model.eval()
    for batch in dataloader:
        subject_id = batch.get("id", None)
        if subject_id is None:
            # fallback: use filename or index
            subject_id = batch["image_meta_dict"]["filename_or_obj"][0]

        inputs = batch["image"].to(opts.device, non_blocking=True)
        labels = batch["label"].to(opts.device, non_blocking=True).squeeze(1)

        if "gin" in opts.method:
            inputs = inputs.repeat_interleave(3, dim=1)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = sliding_window_wrapper(
                inputs,
                model,
                crop_size=opts.crop_size,
                sw_batch_size=opts.batch_size,
            )

        outputs = torch.argmax(outputs, dim=1)

        # one-hot
        pred_oh = torch.nn.functional.one_hot(
            outputs.long(), num_classes=opts.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        label_oh = torch.nn.functional.one_hot(
            labels.long(), num_classes=opts.num_classes
        ).permute(0, 4, 1, 2, 3).float()

        # compute dice for this subject only
        dice_metric.reset()
        dice_metric(pred_oh, label_oh)
        
        d = dice_metric.aggregate()   # shape: (C-1,)
        per_subject.append((subject_id, d.cpu().numpy()))

    # save the per subject results
    per_subject_file = os.path.join(opts.output_dir, "test_per_subject.csv")
    with open(per_subject_file, "w") as f:
        f.write("subject,class,dice\n")
        for subject_id, dice_vec in per_subject:
            for cls_idx in range(1, opts.num_classes):
                cls_name = opts.class_mapping[cls_idx]
                f.write(f"{subject_id},{cls_name},{dice_vec[cls_idx-1]:.4f}\n")

    print(f"Per-subject results saved to {per_subject_file}")

    # save overall test results
    score_file = os.path.join(opts.output_dir, "test_results.csv")
    with open(score_file, "w") as f:
        f.write("class,dice,count\n")
        for class_name, stats in scores["per_class_dice"].items():
            f.write(f"{class_name},{stats['dice']:.4f},{stats['count']}\n")
        f.write(f"mean_dice,{scores['mean_dice']:.4f},\n")
    print(f"Test results saved to {score_file}")



if __name__ == "__main__":
    opts = options.Options()
    args = opts.parse()
    main(args)
