import os

# ==========================================
# 1. 全局基础配置 (Base Configuration)
# ==========================================
BASE_CONFIG = {
    # --- 训练与数据参数 ---
    "batch_size": 1024,
    "epochs": 10,
    "k_folds": 4,
    "lr": 0.01,
    # --- 模型结构参数 ---
    "in_channel": 1,
    "hidden_channel": 1,
    "out_channel": 1,
    "input_size": 252,
    "hidden_size": 252,
    "T": 2,  # 默认扩散步数
    "num_classes": 2,
    # --- 解码器类型 ---
    # 支持: "unet", "segformer_b0", "mobilevit_small"
    "decoder_type": "unet",
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
        "override_config": {"T": 5},  # 默认值，可在命令行动态修改
    },
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

    return final_config, ablation_info["checkpoint"], ablation_info["save_dir"]
