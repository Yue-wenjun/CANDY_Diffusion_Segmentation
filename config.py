import os

# ==========================================
# 1. 全局基础配置 (Base Configuration)
# ==========================================
BASE_CONFIG = {
    # --- 训练与数据参数 ---
    "batch_size": 64,
    "epochs": 11,
    "k_folds": 4,
    "lr": 0.0005,
    # --- 模型结构参数 ---
    "in_channel": 1,
    "hidden_channel": 1,
    "out_channel": 1,
    "input_size": 252,
    "hidden_size": 252,
    "T": 2,  # 默认扩散步数
    "num_classes": 1,
    # --- 解码器类型: "unet" | "segformer_b0" | "mobilevit_small" ---
    "decoder_type": "unet",
    # --- 测试数据目录（干净数据默认；噪声测试在 ABLATION_REGISTRY 里 override）---
    "test_data_dir": "cropped_images",
}


# ==========================================
# 2. 消融实验注册表 (Ablation Registry)
# ==========================================
# 在这里注册你所有的实验变体。
# override_config 会自动覆盖 BASE_CONFIG 里的同名参数。
ABLATION_REGISTRY = {
    "baseline": {
        "checkpoint": "checkpoint/baseline.pth",
        "save_dir": "./imgs/baseline",
        "override_config": {},
    },
    # 【新增】SegFormer 解码器变体
    "segformer": {
        "checkpoint": "checkpoint/segformer.pth",
        "save_dir": "./imgs/segformer",
        "override_config": {"decoder_type": "segformer_b0"},  # 一键切换为 SegFormer
    },
    # 【新增】MobileViT 解码器变体
    "mobilevit": {
        "checkpoint": "checkpoint/mobilevit.pth",
        "save_dir": "./imgs/mobilevit",
        "override_config": {"decoder_type": "mobilevit_small"},  # 一键切换为 MobileViT
    },
    "no_skip": {
        "checkpoint": "checkpoint/no_skip.pth",
        "save_dir": "./imgs/no_skip",
        "override_config": {},
    },
    "sde": {
        "checkpoint": "checkpoint/sde.pth",
        "save_dir": "./imgs/sde",
        "override_config": {},
    },
    "simple_cnn": {
        "checkpoint": "checkpoint/simple_cnn.pth",
        "save_dir": "./imgs/simple_cnn",
        "override_config": {"hidden_channel": 64},
    },
    "simple_decoder": {
        "checkpoint": "checkpoint/simple_decoder.pth",
        "save_dir": "./imgs/simple_decoder",
        "override_config": {},
    },
    "adjust_steps": {
        "checkpoint": "checkpoint/adjust_steps.pth",
        "save_dir": "./imgs/adjust_steps",
        "override_config": {"T": 5},
    },
    # DDPM 基线：用高斯噪声前向过程替换 CANDY，其余结构相同，验证 CANDY 前向的优越性
    "ddpm": {
        "checkpoint": "checkpoint/ddpm.pth",
        "save_dir": "./imgs/ddpm",
        "override_config": {},
    },
    # ==========================================
    # 噪声鲁棒性测试（复用训练权重，切换测试数据目录）
    # 用法: python test.py baseline_0dB -k 4 -f <best_fold>
    # 做噪声测试的模型: baseline(T=2) / adjust_steps(T=5) / segformer / mobilevit
    # ==========================================
    "baseline_0dB":  {"checkpoint": "checkpoint/baseline.pth",      "save_dir": "./imgs/baseline_0dB",      "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_0dB"}},
    "baseline_10dB": {"checkpoint": "checkpoint/baseline.pth",      "save_dir": "./imgs/baseline_10dB",     "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_10dB"}},
    "baseline_20dB": {"checkpoint": "checkpoint/baseline.pth",      "save_dir": "./imgs/baseline_20dB",     "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_20dB"}},

    "adjust_steps_0dB":  {"checkpoint": "checkpoint/adjust_steps.pth", "save_dir": "./imgs/adjust_steps_0dB",  "override_config": {"T": 5, "test_data_dir": "cropped_noised_data/cropped_noised_0dB"}},
    "adjust_steps_10dB": {"checkpoint": "checkpoint/adjust_steps.pth", "save_dir": "./imgs/adjust_steps_10dB", "override_config": {"T": 5, "test_data_dir": "cropped_noised_data/cropped_noised_10dB"}},
    "adjust_steps_20dB": {"checkpoint": "checkpoint/adjust_steps.pth", "save_dir": "./imgs/adjust_steps_20dB", "override_config": {"T": 5, "test_data_dir": "cropped_noised_data/cropped_noised_20dB"}},

    "segformer_0dB":  {"checkpoint": "checkpoint/segformer.pth",    "save_dir": "./imgs/segformer_0dB",     "override_config": {"decoder_type": "segformer_b0",    "test_data_dir": "cropped_noised_data/cropped_noised_0dB"}},
    "segformer_10dB": {"checkpoint": "checkpoint/segformer.pth",    "save_dir": "./imgs/segformer_10dB",    "override_config": {"decoder_type": "segformer_b0",    "test_data_dir": "cropped_noised_data/cropped_noised_10dB"}},
    "segformer_20dB": {"checkpoint": "checkpoint/segformer.pth",    "save_dir": "./imgs/segformer_20dB",    "override_config": {"decoder_type": "segformer_b0",    "test_data_dir": "cropped_noised_data/cropped_noised_20dB"}},

    "mobilevit_0dB":  {"checkpoint": "checkpoint/mobilevit.pth",    "save_dir": "./imgs/mobilevit_0dB",     "override_config": {"decoder_type": "mobilevit_small", "test_data_dir": "cropped_noised_data/cropped_noised_0dB"}},
    "mobilevit_10dB": {"checkpoint": "checkpoint/mobilevit.pth",    "save_dir": "./imgs/mobilevit_10dB",    "override_config": {"decoder_type": "mobilevit_small", "test_data_dir": "cropped_noised_data/cropped_noised_10dB"}},
    "mobilevit_20dB": {"checkpoint": "checkpoint/mobilevit.pth",    "save_dir": "./imgs/mobilevit_20dB",    "override_config": {"decoder_type": "mobilevit_small", "test_data_dir": "cropped_noised_data/cropped_noised_20dB"}},

    "ddpm_0dB":  {"checkpoint": "checkpoint/ddpm.pth", "save_dir": "./imgs/ddpm_0dB",  "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_0dB"}},
    "ddpm_10dB": {"checkpoint": "checkpoint/ddpm.pth", "save_dir": "./imgs/ddpm_10dB", "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_10dB"}},
    "ddpm_20dB": {"checkpoint": "checkpoint/ddpm.pth", "save_dir": "./imgs/ddpm_20dB", "override_config": {"test_data_dir": "cropped_noised_data/cropped_noised_20dB"}},
}


# ==========================================
# 3. 配置加载工具
# ==========================================
def get_config(model_type, custom_steps=None):
    """获取合并后的完整模型配置以及路径"""
    if model_type not in ABLATION_REGISTRY:
        raise ValueError(
            f"未知的模型类型: {model_type}\n可用类型: {list(ABLATION_REGISTRY.keys())}"
        )

    # 1. 拷贝基础配置
    final_config = BASE_CONFIG.copy()
    ablation_info = ABLATION_REGISTRY[model_type]

    # 2. 覆盖当前实验的专属配置
    if "override_config" in ablation_info:
        final_config.update(ablation_info["override_config"])

    # 3. 处理动态步数调整
    if custom_steps is not None and model_type == "adjust_steps":
        final_config["T"] = custom_steps

    base_name, _ = os.path.splitext(ablation_info["checkpoint"])
    final_config["split_record_template"] = f"{base_name}_fold{{fold}}_val_files.json"

    return final_config, ablation_info["checkpoint"], ablation_info["save_dir"]
