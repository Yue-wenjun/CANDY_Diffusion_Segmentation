import argparse
import os
import subprocess
import datetime
import time
import traceback
from config import BASE_CONFIG


def run_ablation_sweep(model_types, epochs=10, k_folds=4, fold=0, loss_spec="bce", lr=None):
    """
    Load data once, train each model in-process for `epochs` epochs, print IoU table.
    Usage: python run.py --ablation [-e N] [-k K] [-f F] [-l LOSS]
    """
    import torch
    from config import get_config
    from models.models import DiffusionModelWrapper
    from data_loading import get_kfold_dataloaders
    from train import train, val, save_checkpoint
    from main import build_criterion

    print(f"\nLoading data (fold {fold + 1}/{k_folds})...")
    train_loader, val_loader = get_kfold_dataloaders(
        "cropped_images", "cropped_masks",
        BASE_CONFIG["batch_size"], n_splits=k_folds, fold=fold,
    )
    device = torch.device("cuda")
    criterion = build_criterion(loss_spec, device)
    results = {}

    for model_type in model_types:
        print(f"\n{'=' * 50}")
        print(f"Ablation: {model_type}  epochs={epochs}  loss={loss_spec}")
        print(f"{'=' * 50}")

        config, ckpt_template, _ = get_config(model_type, None)
        if lr is not None:
            config["lr"] = lr

        model = DiffusionModelWrapper(config).create_model(model_type).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=config["lr"],
            steps_per_epoch=len(train_loader), epochs=epochs, pct_start=0.1,
        )

        base_name, ext = os.path.splitext(ckpt_template)
        best_ckpt = f"{base_name}_fold{fold + 1}_best{ext}"
        os.makedirs(os.path.dirname(best_ckpt), exist_ok=True)

        best_iou = -1.0
        for epoch in range(epochs):
            train(model, train_loader, optimizer, scheduler, device, epoch,
                  BASE_CONFIG["batch_size"], best_ckpt, criterion)
            m = val(model, val_loader, device, BASE_CONFIG["batch_size"],
                    criterion, verbose=False)
            if m["iou"] > best_iou:
                best_iou = m["iou"]
                save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)
            print(f"  epoch {epoch + 1}/{epochs}  IoU={m['iou']:.4f}  best={best_iou:.4f}")

        results[model_type] = best_iou
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    print(f"\n{'=' * 50}")
    print(f"Ablation summary  loss={loss_spec}  epochs={epochs}  fold={fold + 1}")
    print(f"{'=' * 50}")
    print(f"{'Model':<22} {'Best IoU':>10}")
    print("-" * 34)
    for mt, iou in sorted(results.items(), key=lambda x: -x[1]):
        print(f"{mt:<22} {iou:>10.4f}")


def run_model(model_name, epochs=10, k_folds=4, steps=None):
    """运行单个模型"""
    print(f"\n{'=' * 60}")
    print(f"开始运行: {model_name} 模型")
    print(f"参数: epochs={epochs}, k_folds={k_folds}, steps={steps}")
    print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 60}")

    cmd = ["python", "main.py", model_name, "-e", str(epochs), "-k", str(k_folds)]
    if steps and model_name == "adjust_steps":
        cmd.extend(["-s", str(steps)])

    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"\n{'=' * 60}\n")
        log_file.write(f"模型: {model_name}\n")
        log_file.write(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'=' * 60}\n")
        log_file.flush()

        # 运行命令
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding='utf-8',      # 【关键添加】强制使用 UTF-8 读取
            errors='replace'       # 【关键添加】遇到解析不了的乱码直接替换成问号，绝不崩溃
        )


        # 实时输出到屏幕和文件
        for line in process.stdout:
            print(line, end='')
            log_file.write(line)
            log_file.flush()

        process.wait()


ABLATION_MODELS = [
    # CANDY + decoder
    "baseline",          # CANDY + UNet
    "segformer",         # CANDY + SegFormer
    "mobilevit",         # CANDY + MobileViT
    # Pure decoder (no CANDY) — proves CANDY adds value
    "pure_unet",
    "pure_segformer",
    "pure_mobilevit",
]

SEQUENTIAL_MODELS = [
    # "mobilevit",
    "baseline",
    # "adjust_steps",
]


def main():
    parser = argparse.ArgumentParser(description="run.py — training launcher")
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation sweep (data loaded once, all models in-process)",
    )
    parser.add_argument("-e", "--epochs", type=int, default=BASE_CONFIG["epochs"])
    parser.add_argument("-k", "--kfolds", type=int, default=BASE_CONFIG["k_folds"])
    parser.add_argument("-f", "--fold",   type=int, default=0)
    parser.add_argument("-l", "--loss",   type=str, default="bce")
    args = parser.parse_args()

    if args.ablation:
        run_ablation_sweep(
            ABLATION_MODELS, epochs=args.epochs, k_folds=args.kfolds,
            fold=args.fold, loss_spec=args.loss,
        )
        return

    # ── Sequential subprocess runs ─────────────────────────────────────────────
    total_start = datetime.datetime.now()
    print(f"总开始时间: {total_start.strftime('%Y-%m-%d %H:%M:%S')}")

    for model in SEQUENTIAL_MODELS:
        try:
            steps = 5 if model == "adjust_steps" else None
            run_model(model, epochs=args.epochs, k_folds=args.kfolds, steps=steps)
            time.sleep(2)
        except Exception as e:
            print(f"\n运行模型 {model} 时出错: {e}")
            traceback.print_exc()

    total_end = datetime.datetime.now()
    print(f"\n所有模型运行完成!  总运行时间: {total_end - total_start}")


if __name__ == "__main__":
    main()