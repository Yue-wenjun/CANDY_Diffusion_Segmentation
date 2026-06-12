import argparse
import torch
import os
import time
from monai.losses import DiceLoss

from models.models import DiffusionModelWrapper
from data_loading import get_dataloaders, get_kfold_dataloaders
from train import train, val, save_checkpoint, load_checkpoint
from utils import app

# 【引入配置】
from config import BASE_CONFIG, get_config, ABLATION_REGISTRY


def build_criterion(loss_spec: str, device):
    """
    Parse a loss spec string into a callable criterion.

    Formats:
      dice              → DiceLoss(sigmoid=True)
      bce               → BCEWithLogitsLoss(pos_weight=10)
      bce:N             → BCEWithLogitsLoss(pos_weight=N)
      dice+bce:W        → DiceLoss + W * BCEWithLogitsLoss(pos_weight=10)
      dice+bce:W:N      → DiceLoss + W * BCEWithLogitsLoss(pos_weight=N)
    """
    spec = loss_spec.strip().lower()

    if spec == "dice":
        return DiceLoss(sigmoid=True)

    if spec.startswith("bce"):
        parts = spec.split(":")
        pw = float(parts[1]) if len(parts) > 1 else 10.0
        return torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], device=device)
        )

    if spec.startswith("dice+bce"):
        parts = spec.split(":")
        bce_w = float(parts[1]) if len(parts) > 1 else 0.2
        pw    = float(parts[2]) if len(parts) > 2 else 10.0
        _dice = DiceLoss(sigmoid=True)
        _bce  = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw], device=device)
        )
        # BCE drives optimization (anchors logit scale), Dice is a shape regularizer.
        # bce_w is the Dice weight; BCE weight is always 1 to keep loss on BCE scale.
        return lambda pred, tgt: bce_w * _dice(pred, tgt) + _bce(pred, tgt)

    raise ValueError(
        f"Unknown loss spec '{loss_spec}'. "
        "Use: dice | bce | bce:N | dice+bce:W | dice+bce:W:N"
    )


class DiffusionCLI:
    def __init__(self):
        self.device = torch.device("cuda")

    def run_pipeline(
        self, model_type, num_epochs=1, custom_steps=None, k_folds=1, current_fold=0,
        loss_spec="bce", lr=None,
    ):
        # 1. 直接通过 config.py 获取所有参数
        config, checkpoint_path_template, save_dir = get_config(
            model_type, custom_steps
        )
        if lr is not None:
            config["lr"] = lr

        print(f"\n=== Running {model_type} model ===")
        if model_type == "adjust_steps":
            print(f"Current diffusion steps: {config['T']}")

        if k_folds > 1:
            print(f"K-Fold Cross Validation: Fold {current_fold + 1}/{k_folds}")

        # Data loading
        if k_folds > 1:
            # Use k-fold cross validation
            train_loader, val_loader = get_kfold_dataloaders(
                "cropped_images",
                "cropped_masks",
                BASE_CONFIG["batch_size"],
                n_splits=k_folds,
                fold=current_fold,
            )
            # For k-fold, we don't use test set during training
            test_loader = None
        else:
            # Use standard train/val/test split
            train_loader, val_loader, test_loader = get_dataloaders(
                "cropped_images",
                "cropped_masks",
                BASE_CONFIG["batch_size"],
                0.05,
                0.05,
            )

        model_wrapper = DiffusionModelWrapper(config)
        model = model_wrapper.create_model(model_type).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        criterion = build_criterion(loss_spec, self.device)
        print(f"Loss: {loss_spec}  LR: {config['lr']}")

        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config["lr"],
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            pct_start=0.1,
        )

        # Adjust checkpoint path for k-fold
        base_checkpoint_path = checkpoint_path_template
        base_name, ext = os.path.splitext(base_checkpoint_path)
        # For adjust_steps, embed T in the name to avoid collisions across different step counts
        step_suffix = f"_T{config['T']}" if model_type == "adjust_steps" else ""
        checkpoint_path = f"{base_name}{step_suffix}_fold{current_fold + 1}{ext}"

        # Checkpoint handling
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(checkpoint_path)

        best_checkpoint_path = f"{base_name}{step_suffix}_fold{current_fold + 1}_best{ext}"

        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path} (epoch {start_epoch})")
        else:
            start_epoch = 0
            print("No checkpoint found, initializing new model")

        # Training phase
        print(f"\nTraining started (Total {num_epochs} epochs)...")
        best_iou = -1.0
        best_val_loss, best_dice, best_proportion = None, None, None
        for epoch in range(start_epoch, start_epoch + num_epochs):
            start_time = time.time()
            train_loss = train(
                model,
                train_loader,
                optimizer,
                scheduler,
                self.device,
                epoch,
                BASE_CONFIG["batch_size"],
                checkpoint_path,
                criterion,
            )
            val_metrics = val(
                model,
                val_loader,
                self.device,
                BASE_CONFIG["batch_size"],
                criterion,
            )
            val_loss = val_metrics["loss"]
            iou = val_metrics["iou"]
            dice = val_metrics["dice"]
            proportion = val_metrics["proportion"]

            if iou > best_iou:
                best_iou = iou
                best_val_loss, best_dice, best_proportion = val_loss, dice, proportion
                save_checkpoint(model, optimizer, scheduler, epoch, best_checkpoint_path)
                print(f"Best model updated (IoU={best_iou:.4f}) → {best_checkpoint_path}")

        # Testing phase (only for non-k-fold mode)
        if k_folds <= 1 and test_loader is not None:
            print(f"\nTesting...")
            os.makedirs(save_dir, exist_ok=True)
            app(
                model,
                test_loader,
                self.device,
                BASE_CONFIG["batch_size"],
                save_dir,
            )

            print(f"\n=== Execution completed ===")
            print(f"Test results: {save_dir}")
            print(f"Final checkpoint: {checkpoint_path}")
        else:
            print(f"\n=== Fold {current_fold + 1} completed ===")
            print(f"Best IoU: {best_iou:.4f}  Dice: {best_dice:.4f}")
            print(f"Best checkpoint: {best_checkpoint_path}")
            print(f"Final checkpoint: {checkpoint_path}")

        return best_val_loss, best_iou, best_dice, best_proportion

    def eval_thresh_sweep(
        self, model_type, custom_steps=None, k_folds=1, current_fold=0,
        loss_spec="bce", thresholds=None,
    ):
        """Evaluate a saved best checkpoint across a range of thresholds (no training)."""
        if thresholds is None:
            thresholds = [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]

        config, checkpoint_path_template, _ = get_config(model_type, custom_steps)

        if k_folds > 1:
            _, val_loader = get_kfold_dataloaders(
                "cropped_images", "cropped_masks",
                BASE_CONFIG["batch_size"], n_splits=k_folds, fold=current_fold,
            )
        else:
            _, val_loader, _ = get_dataloaders(
                "cropped_images", "cropped_masks",
                BASE_CONFIG["batch_size"], 0.05, 0.05,
            )

        model_wrapper = DiffusionModelWrapper(config)
        model = model_wrapper.create_model(model_type).to(self.device)

        base_name, ext = os.path.splitext(checkpoint_path_template)
        step_suffix = f"_T{config['T']}" if model_type == "adjust_steps" else ""
        best_ckpt = f"{base_name}{step_suffix}_fold{current_fold + 1}_best{ext}"

        if not os.path.exists(best_ckpt):
            print(f"No best checkpoint found: {best_ckpt}")
            return

        load_checkpoint(model, None, None, best_ckpt)
        criterion = build_criterion(loss_spec, self.device)

        print(f"\nThreshold sweep on {best_ckpt}")
        print(f"{'Threshold':>12}  {'IoU':>8}  {'Dice':>8}")
        print("-" * 36)
        best_t, best_iou = thresholds[0], -1.0
        for t in thresholds:
            m = val(model, val_loader, self.device, BASE_CONFIG["batch_size"],
                    criterion, thresh=t, verbose=False)
            marker = " ←" if m["iou"] > best_iou else ""
            if m["iou"] > best_iou:
                best_iou, best_t = m["iou"], t
            print(f"{t:>12.2f}  {m['iou']:>8.4f}  {m['dice']:>8.4f}{marker}")
        print(f"\nBest threshold: {best_t:.2f}  IoU={best_iou:.4f}")

    def run_kfold_cross_validation(
        self, model_type, num_epochs=1, custom_steps=None, k_folds=5, loss_spec="bce", lr=None,
    ):
        """Run k-fold cross validation"""
        print(f"\nStarting {k_folds}-Fold Cross Validation for {model_type} model")

        # 存储每折结果
        fold_results = []
        all_val_losses, all_ious, all_dices, all_props = [], [], [], []

        for fold in range(k_folds):
            print(f"\n{'=' * 50}")
            print(f"Fold {fold + 1}/{k_folds}")
            print(f"{'=' * 50}")

            try:
                # 接收 run_pipeline() 的返回结果
                val_loss, iou, dice, proportion = self.run_pipeline(
                    model_type=model_type,
                    num_epochs=num_epochs,
                    custom_steps=custom_steps,
                    k_folds=k_folds,
                    current_fold=fold,
                    loss_spec=loss_spec,
                    lr=lr,
                )

                # 保存结果
                all_val_losses.append(val_loss)
                all_ious.append(iou)
                all_dices.append(dice)
                all_props.append(proportion)

                # 每折输出
                result_text = (
                    "Fold "
                    + str(fold)
                    + ": "
                    + "Val_Loss="
                    + str(val_loss)
                    + ", "
                    + "IoU="
                    + str(iou)
                    + ", "
                    + "Dice="
                    + str(dice)
                    + ", "
                    + "Proportion="
                    + str(proportion)
                )
                print(result_text)
                fold_results.append(result_text)

            except Exception as e:
                import traceback

                error_details = traceback.format_exc()
                print(f"Detailed error for fold {fold + 1}:")
                print(error_details)
                fold_results.append(f"Fold {fold + 1}: Failed - {str(e)}")

        # ========== Summary Section ==========

        print(f"\n{'=' * 50}")
        print("K-Fold Cross Validation Summary")
        print(f"{'=' * 50}")

        for result in fold_results:
            print(result)

        # 如果全部成功则输出平均指标
        if len(all_ious) == k_folds:
            mean_val = sum(all_val_losses) / k_folds
            mean_iou = sum(all_ious) / k_folds
            mean_dice = sum(all_dices) / k_folds
            mean_prop = sum(all_props) / k_folds

            print(f"\nAverage Results Across {k_folds} Folds:")
            print(f"Mean Val_Loss = {mean_val:.4f}")
            print(f"Mean IoU      = {mean_iou:.4f}")
            print(f"Mean Dice     = {mean_dice:.4f}")
            print(f"Mean Proportion = {mean_prop:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Model Training and Testing System\n"
        "Available models: baseline, no_skip, sde, simple_cnn, simple_decoder, adjust_steps",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model_type",
        choices=list(ABLATION_REGISTRY.keys()),
        help="Select model type",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "-s", "--steps", type=int, help="For adjust_steps only: custom diffusion steps"
    )
    parser.add_argument(
        "-k",
        "--kfolds",
        type=int,
        default=1,
        help="Number of folds for k-fold cross validation (default: 1, no cross validation)",
    )
    parser.add_argument(
        "-f",
        "--fold",
        type=int,
        default=None,
        help="Run a single specific fold (0-indexed). Requires -k. e.g. -k 4 -f 0 runs only fold 1.",
    )
    parser.add_argument(
        "-l",
        "--loss",
        type=str,
        default="bce",
        help=(
            "Loss function spec (default: bce). Options:\n"
            "  dice            → DiceLoss(sigmoid=True)\n"
            "  bce             → BCEWithLogitsLoss(pos_weight=10)\n"
            "  bce:N           → BCEWithLogitsLoss(pos_weight=N)\n"
            "  dice+bce:W      → DiceLoss + W*BCE(pos_weight=10)\n"
            "  dice+bce:W:N    → DiceLoss + W*BCE(pos_weight=N)\n"
        ),
    )

    parser.add_argument(
        "--eval-thresh",
        action="store_true",
        help="Sweep thresholds on the saved best checkpoint (no training).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate (default: use config). Use ~5e-5 for dice+bce to fix scale mismatch.",
    )

    args = parser.parse_args()

    # Parameter validation
    if args.model_type == "adjust_steps" and not args.steps:
        parser.error("adjust_steps requires -s parameter (e.g., -s 5)")
    if args.fold is not None and args.kfolds <= 1:
        parser.error("-f requires -k > 1")

    try:
        cli = DiffusionCLI()

        if args.eval_thresh:
            fold = args.fold if args.fold is not None else 0
            cli.eval_thresh_sweep(
                args.model_type,
                custom_steps=args.steps,
                k_folds=args.kfolds,
                current_fold=fold,
                loss_spec=args.loss,
            )
        elif args.fold is not None:
            # Run a single specified fold
            cli.run_pipeline(
                args.model_type,
                num_epochs=args.epochs,
                custom_steps=args.steps,
                k_folds=args.kfolds,
                current_fold=args.fold,
                loss_spec=args.loss,
                lr=args.lr,
            )
        elif args.kfolds > 1:
            # Run k-fold cross validation
            cli.run_kfold_cross_validation(
                args.model_type,
                num_epochs=args.epochs,
                custom_steps=args.steps,
                k_folds=args.kfolds,
                loss_spec=args.loss,
                lr=args.lr,
            )
        else:
            # Run standard pipeline
            cli.run_pipeline(
                args.model_type, num_epochs=args.epochs, custom_steps=args.steps,
                loss_spec=args.loss, lr=args.lr,
            )

    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)
