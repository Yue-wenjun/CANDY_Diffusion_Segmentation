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

MODELS_TO_TEST = [
    "baseline", "ddpm", "mobilevit", "no_skip",
    "sde", "segformer", "simple_cnn", "simple_decoder",
    "adjust_steps",
]
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
_CSV_FIELDS = ["Model", "Fold", "Condition", "Loss", "IoU", "Dice", "Proportion"]


# ==============================================================================
# ⬇️ 以下为核心逻辑代码 ⬇️
# ==============================================================================

def _csv_path():
    return os.path.join(OUTPUT_ROOT, CSV_FILENAME)


def _load_existing():
    """加载已有 CSV，返回 (行列表, 已完成key集合)，支持断点续跑。"""
    rows, done = [], set()
    path = _csv_path()
    if not os.path.exists(path):
        return rows, done
    with open(path, newline='', encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows.append(row)
            done.add((row["Model"], int(row["Fold"]), row["Condition"]))
    print(f"检测到已有记录 {len(rows)} 条，已完成项将自动跳过。")
    return rows, done


def _save_csv(results):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(_csv_path(), mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)


def run_all_tests():
    device = torch.device("cuda")

    print(f"\n{'=' * 60}")
    print(f"🚀 开始一键多折测试 (包含原图与多噪声鲁棒性) 🚀")
    print(f"测试模型: {MODELS_TO_TEST}")
    print(f"包含干净数据: {TEST_CLEAN_DATA}")
    print(f"噪声等级: {NOISE_LEVELS} dB")
    print(f"{'=' * 60}")

    all_results, done_keys = _load_existing()

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
                key = (model_type, fold, condition_name)

                if key in done_keys:
                    print(f"【跳过(已完成)】{model_type} Fold {fold} @ {condition_name}")
                    continue

                checkpoint_path = CHECKPOINT_TEMPLATE.format(model=model_type, fold=fold)
                if not os.path.exists(checkpoint_path):
                    print(f"【跳过(无权重)】{checkpoint_path}")
                    continue

                print(f"\n--- 测试 {model_type} - Fold {fold} @ {condition_name} ---")

                model_wrapper = DiffusionModelWrapper(config)
                model = model_wrapper.create_model(model_type).to(device)
                load_checkpoint(model, None, None, checkpoint_path)
                model.eval()

                save_dir = os.path.join(OUTPUT_ROOT, model_type, f"fold{fold}_{condition_name}")
                os.makedirs(save_dir, exist_ok=True)

                metrics = app(model, test_loader, device, BASE_CONFIG["batch_size"], save_dir)

                del model
                torch.cuda.empty_cache()

                if metrics:
                    all_results.append({
                        "Model":      model_type,
                        "Fold":       fold,
                        "Condition":  condition_name,
                        "Loss":       metrics["loss"],
                        "IoU":        metrics["iou"],
                        "Dice":       metrics["dice"],
                        "Proportion": metrics["proportion"],
                    })
                    done_keys.add(key)
                    _save_csv(all_results)
                    print(f">>> CSV 已保存 ({len(all_results)} 条): {_csv_path()}")

    print(f"\n{'=' * 60}")
    print(f"🎉 所有测试完成！结果: {_csv_path()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n【发生严重错误】")
        import traceback
        traceback.print_exc()
