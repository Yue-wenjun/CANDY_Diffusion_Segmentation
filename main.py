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


class DiffusionCLI:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_pipeline(
        self, model_type, num_epochs=1, custom_steps=None, k_folds=1, current_fold=0
    ):
        # 1. 直接通过 config.py 获取所有参数
        config, checkpoint_path_template, save_dir = get_config(
            model_type, custom_steps
        )

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

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        criterion = DiceLoss(sigmoid=True)

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
        checkpoint_path = f"{base_name}3_fold{current_fold + 1}{ext}"

        # Checkpoint handling
        checkpoint_dir = os.path.dirname(checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)

        print(checkpoint_path)

        if os.path.exists(checkpoint_path):
            start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
            print(f"Loaded checkpoint: {checkpoint_path} (epoch {start_epoch})")
        else:
            start_epoch = 0
            print("No checkpoint found, initializing new model")

        # Training phase
        print(f"\nTraining started (Total {num_epochs} epochs)...")
        last_val_loss, last_iou, last_dice, last_proportion = None, None, None, None
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
                checkpoint_path,
            )
            val_loss = val_metrics["loss"]
            iou = val_metrics["iou"]
            dice = val_metrics["dice"]
            proportion = val_metrics["proportion"]

            last_val_loss, last_iou, last_dice, last_proportion = (
                val_loss,
                iou,
                dice,
                proportion,
            )

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
            print(f"Final checkpoint: {checkpoint_path}")

        return last_val_loss, last_iou, last_dice, last_proportion

    def run_kfold_cross_validation(
        self, model_type, num_epochs=1, custom_steps=None, k_folds=5
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

    args = parser.parse_args()

    # Parameter validation
    if args.model_type == "adjust_steps" and not args.steps:
        parser.error("adjust_steps requires -s parameter (e.g., -s 5)")

    try:
        cli = DiffusionCLI()

        if args.kfolds > 1:
            # Run k-fold cross validation
            cli.run_kfold_cross_validation(
                args.model_type,
                num_epochs=args.epochs,
                custom_steps=args.steps,
                k_folds=args.kfolds,
            )
        else:
            # Run standard pipeline
            cli.run_pipeline(
                args.model_type, num_epochs=args.epochs, custom_steps=args.steps
            )

    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)
