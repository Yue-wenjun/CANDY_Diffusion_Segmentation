import argparse
import torch
import os

from models.models import DiffusionModelWrapper
from data_loading import get_kfold_test_dataloader
from train import load_checkpoint
from utils import app
from config import BASE_CONFIG, get_config, ABLATION_REGISTRY


def get_test_dataloader(image_dir, mask_dir, batch_size, **_):
    """
    Always returns the fixed held-out last 15% test set.
    This is the same slice used by get_kfold_dataloaders, ensuring no data leakage.
    Extra keyword args (**_) are accepted but ignored for backward compatibility.
    """
    return get_kfold_test_dataloader(image_dir, mask_dir, batch_size, test_ratio=0.15)


def test_model(model_type, checkpoint_path=None, custom_steps=None, k_folds=1, fold=1, data_dir=None):
    device = torch.device("cuda")
    print(f"\n=== Testing {model_type} model on {device} ===")

    # 1. 获取模型配置
    config, default_checkpoint, save_dir = get_config(model_type, custom_steps)

    # 2. 确定权重路径与 JSON 记录路径
    if checkpoint_path is None:
        if k_folds > 1:
            # 推导 K折 的默认权重路径 (假设命名规则与 main.py 保持一致)
            base_name, ext = os.path.splitext(default_checkpoint)
            checkpoint_path = f"{base_name}_fold{fold}{ext}"
        else:
            checkpoint_path = default_checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到权重文件: {checkpoint_path}\n请先训练模型或检查路径是否输入正确。")

    print(f"即将加载权重: {checkpoint_path}")

    # 推导 JSON 记录路径
    split_record_path = None
    if k_folds > 1:
        # 假设 JSON 文件与权重文件在同级目录，且命名规范为 base_foldX_val_files.json
        base_ckpt_name, _ = os.path.splitext(default_checkpoint)
        split_record_path = f"{base_ckpt_name}_fold{fold}_val_files.json"
        print(f"预期的数据划分记录路径: {split_record_path}")

    # 3. 加载数据（优先级：命令行 -d > config["test_data_dir"] > 默认干净数据）
    image_dir = data_dir if data_dir else config.get("test_data_dir", "cropped_images")
    mask_dir = "cropped_masks"

    test_loader = get_test_dataloader(
        image_dir=image_dir,
        mask_dir=mask_dir,
        batch_size=BASE_CONFIG["batch_size"],
        k_folds=k_folds,
        fold=fold,
        split_record_path=split_record_path
    )

    if test_loader is None or len(test_loader.dataset) == 0:
        raise ValueError("Test loader is empty. 请检查数据集路径或 JSON 划分记录。")

    # 4. 初始化模型
    model_wrapper = DiffusionModelWrapper(config)
    model = model_wrapper.create_model(model_type).to(device)

    # 5. 加载预训练权重
    load_checkpoint(model, None, None, checkpoint_path)
    model.eval()

    # 6. 运行测试并保存结果
    # 如果测试的是不同折或加噪数据，可以在 save_dir 后面加个后缀防止覆盖
    if k_folds > 1:
        save_dir = f"{save_dir}_fold{fold}"
    if data_dir:
        noise_level = os.path.basename(os.path.normpath(data_dir))
        save_dir = f"{save_dir}_{noise_level}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n开始在测试集上进行推理与指标评估...")
    print(f"可视化结果图将保存在目录: {save_dir}")

    # 执行测试流程
    app(
        model,
        test_loader,
        device,
        BASE_CONFIG["batch_size"],
        save_dir,
    )

    print("\n=== 测试流程全部完成 ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Diffusion Model Independent Testing Script",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model_type",
        choices=list(ABLATION_REGISTRY.keys()),
        help="Select model type to test",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str,
        default=None,
        help="指定 .pth 权重文件的路径。如果不填，默认使用推导出的路径。"
    )
    parser.add_argument(
        "-s", "--steps",
        type=int,
        help="For adjust_steps only: custom diffusion steps"
    )
    parser.add_argument(
        "-k", "--kfolds",
        type=int,
        default=1,
        help="是否使用 K-Fold 模式进行测试 (输入 4 代表使用 4 折模式，默认 1 代表固定后 15%)"
    )
    parser.add_argument(
        "-f", "--fold",
        type=int,
        default=1,
        help="当使用 K-Fold 模式时，指定测试第几折 (默认: 1)"
    )
    parser.add_argument(
        "-d", "--data_dir",
        type=str,
        default=None,
        help="指定测试集图片目录 (例如加噪数据集路径)。如果不填，默认使用 cropped_noised_data/cropped_noised_0dB"
    )

    args = parser.parse_args()

    # 参数校验
    if args.model_type == "adjust_steps" and not args.steps:
        parser.error("adjust_steps 模型必须提供 -s 参数 (例如: -s 5)")

    try:
        test_model(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            custom_steps=args.steps,
            k_folds=args.kfolds,
            fold=args.fold,
            data_dir=args.data_dir
        )
    except Exception as e:
        print(f"\n测试发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)