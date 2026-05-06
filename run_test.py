import os
import torch
import csv

from models.models import DiffusionModelWrapper
from data_loading import get_test_only_dataloader
from train import load_checkpoint
from utils import app
from config import BASE_CONFIG, get_config

# ==============================================================================
# 🔴 【一键运行配置区】 🔴
# ==============================================================================

MODELS_TO_TEST = ["baseline"]
K_FOLDS = 4
ADJUST_STEPS_VAL = 5

# 1. 权重路径配置
CHECKPOINT_TEMPLATE = "checkpoint/{model}_fold{fold}.pth"

# 2. 数据路径配置
TEST_CLEAN_DATA = True
CLEAN_IMAGE_PATH = "cropped_images"

NOISE_LEVELS = [0, 5, 10, 15, 20, 25, 30]
IMAGE_PATH_TEMPLATE = "cropped_noised_data/cropped_noised_{db}dB"

MASK_PATH = "cropped_masks"

# 3. 输出路径配置
OUTPUT_ROOT = "noise_test_results"
CSV_FILENAME = "多折多噪声测试指标汇总.csv"


# ==============================================================================
# ⬇️ 以下为核心逻辑代码 ⬇️
# ==============================================================================

def run_all_tests():
    device = torch.device("cuda")

    print(f"\n{'=' * 60}")
    print(f"🚀 开始一键多折测试 (包含原图与多噪声鲁棒性) 🚀")
    print(f"测试模型: {MODELS_TO_TEST}")
    print(f"包含干净数据: {TEST_CLEAN_DATA}")
    print(f"噪声等级: {NOISE_LEVELS} dB")
    print(f"{'=' * 60}")

    all_results = []

    # 构建所有测试条件
    test_conditions = []
    if TEST_CLEAN_DATA:
        test_conditions.append(("Clean", CLEAN_IMAGE_PATH))
    for db in NOISE_LEVELS:
        test_conditions.append((f"{db}dB", IMAGE_PATH_TEMPLATE.format(db=db)))

    # 外层循环：按数据集遍历，每个数据集只加载一次
    for condition_name, img_dir in test_conditions:
        if not os.path.exists(img_dir) or not os.path.exists(MASK_PATH):
            print(f"【跳过】找不到路径: {img_dir} 或 {MASK_PATH}")
            continue

        print(f"\n{'=' * 60}")
        print(f"📂 加载数据集: {condition_name}  ({img_dir})")
        test_loader = get_test_only_dataloader(
            img_dir, MASK_PATH, BASE_CONFIG["batch_size"]
        )
        if len(test_loader.dataset) == 0:
            print(f"【跳过】数据集为空: {img_dir}")
            continue

        # 内层循环：对当前数据集跑完所有模型的所有 fold
        for model_type in MODELS_TO_TEST:
            custom_steps = ADJUST_STEPS_VAL if model_type == "adjust_steps" else None
            config, _, _ = get_config(model_type, custom_steps)

            for fold in range(1, K_FOLDS + 1):
                checkpoint_path = CHECKPOINT_TEMPLATE.format(model=model_type, fold=fold)

                if not os.path.exists(checkpoint_path):
                    print(f"【跳过】找不到 Fold {fold} 的权重: {checkpoint_path}")
                    continue

                print(f"\n--- 测试 {model_type} - Fold {fold} @ {condition_name} ---")

                model_wrapper = DiffusionModelWrapper(config)
                model = model_wrapper.create_model(model_type).to(device)
                load_checkpoint(model, None, None, checkpoint_path)
                model.eval()

                save_dir = os.path.join(OUTPUT_ROOT, model_type, f"fold{fold}_{condition_name}")
                os.makedirs(save_dir, exist_ok=True)

                metrics = app(
                    model,
                    test_loader,
                    device,
                    BASE_CONFIG["batch_size"],
                    save_dir,
                )

                if metrics:
                    all_results.append({
                        "Model": model_type,
                        "Fold": fold,
                        "Condition": condition_name,
                        "Loss": metrics["loss"],
                        "IoU": metrics["iou"],
                        "Dice": metrics["dice"],
                        "Proportion": metrics["proportion"]
                    })

                del model
                torch.cuda.empty_cache()

    # 导出 CSV
    if all_results:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        csv_path = os.path.join(OUTPUT_ROOT, CSV_FILENAME)

        with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
            fieldnames = ["Model", "Fold", "Condition", "Loss", "IoU", "Dice", "Proportion"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'=' * 60}")
        print(f"🎉 所有测试完成！")
        print(f"📊 核心评估指标已成功导出至: {csv_path}")
        print(f"{'=' * 60}")
    else:
        print("\n⚠️ 未收集到任何测试数据，请检查前面是否有路径报错。")


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n【发生严重错误】")
        import traceback
        traceback.print_exc()
